import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple
import math
from dataclasses import dataclass
from torch.cuda.amp import autocast, GradScaler
from EnhancedSGD import EnhancedSGD

# ----------------------------
# Data Preparation
# ----------------------------

class ByteDataset(Dataset):
    def __init__(self, data: bytes, context_size: int, stride: int = 1):
        self.data = np.frombuffer(data, dtype=np.uint8)
        self.context_size = context_size
        self.stride = stride

    def __len__(self):
        return (len(self.data) - self.context_size) // self.stride

    def __getitem__(self, idx):
        offset = idx * self.stride
        context = self.data[offset:offset + self.context_size]
        target = self.data[offset + self.context_size]
        return torch.from_numpy(context).long(), torch.tensor(target).long()

def prepare_dataloader(batch_size: int, context_size: int, num_workers: int = 4, device: str = 'cpu'):
    # Load text data from file
    with open('data/wikitext_train.csv', 'r', encoding='utf-8') as f:
        text = f.read()
    byte_data = text.encode('utf-8')
    
    dataset = ByteDataset(byte_data, context_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=device == 'cuda'
    )
    return dataloader

# ----------------------------
# Model Definition
# ----------------------------

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, attention_mask=None):
        # Self-attention with optional mask
        attn_output, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(1, max_len, hidden_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ByteLatentTransformer(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_size: int = 256, 
                 num_layers: int = 4, num_heads: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.context_size = None  # Will be set during forward pass
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout_rate)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)
        
        # Initialize parameters with better defaults
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1/math.sqrt(2))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, x, attention_mask=None):
        self.context_size = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
            
        x = self.norm(x)
        logits = self.output(x)
        return logits

# ----------------------------
# Training Utilities
# ----------------------------

@dataclass
class SamplerConfig:
    low_entropy_threshold: float = 0.3
    medium_entropy_threshold: float = 1.2
    high_entropy_threshold: float = 2.5

def calculate_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Calculate entropy for each sample in the batch."""
    return -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)

def entropy_based_sampling(logits: torch.Tensor, config: SamplerConfig) -> torch.Tensor:
    """Sample next tokens based on entropy thresholds."""
    probs = F.softmax(logits, dim=-1)
    entropy = calculate_entropy(probs)
    
    # Initialize output tensor
    sampled = torch.zeros_like(entropy, dtype=torch.long)
    
    # Low entropy: greedy sampling
    low_mask = entropy < config.low_entropy_threshold
    sampled[low_mask] = torch.argmax(probs[low_mask], dim=-1)
    
    # Medium entropy: top-k sampling
    med_mask = (entropy >= config.low_entropy_threshold) & (entropy < config.medium_entropy_threshold)
    if med_mask.any():
        top_k = 10
        top_k_probs, top_k_indices = torch.topk(probs[med_mask], k=top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        sampled[med_mask] = top_k_indices[torch.arange(top_k_indices.size(0)), 
                                        torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)]
    
    # High entropy: random sampling
    high_mask = entropy >= config.medium_entropy_threshold
    if high_mask.any():
        sampled[high_mask] = torch.multinomial(probs[high_mask], num_samples=1).squeeze(-1)
    
    return sampled

# ----------------------------
# Training and Generation
# ----------------------------

def train_model(
    model: ByteLatentTransformer,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: str,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 1000,
    max_grad_norm: float = 1.0
):
    model = model.to(device)
    
    # Initialize EnhancedSGD optimizer
    optimizer = EnhancedSGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        smoothing_factor=0.1,
        apply_noise=True,
        adaptive_momentum=True,
        gradient_centering=True,
        gradient_clipping=True
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Training loop
    total_steps = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (context, target) in enumerate(dataloader):
            context, target = context.to(device), target.to(device)
            
            # Mixed precision training
            with autocast():
                logits = model(context)
                loss = F.cross_entropy(logits[:, -1, :], target)
                loss = loss / gradient_accumulation_steps
            
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            
            # Accumulate gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                total_steps += 1
                
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')
    
    return model

def generate_text(
    model: ByteLatentTransformer,
    seed_text: str,
    length: int,
    config: SamplerConfig,
    device: str = 'cuda',
    temperature: float = 0.8
):
    model.eval()
    generated = list(seed_text.encode('utf-8'))
    
    with torch.no_grad():
        for _ in range(length):
            # Take the last context_size bytes
            context = torch.tensor([generated[-model.context_size:]], dtype=torch.long).to(device)
            logits = model(context)
            last_logits = logits[:, -1, :]
            
            # Sample next token based on entropy
            sampled_token = entropy_based_sampling(last_logits, config)
            generated.append(int(sampled_token))
    
    return bytes(generated).decode('utf-8', errors='replace')

# ----------------------------
# Main Demo Function
# ----------------------------

def main():
    # Configuration
    context_size = 8
    batch_size = 128
    epochs = 5
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")

    # Prepare DataLoader
    dataloader = prepare_dataloader(batch_size=batch_size, context_size=context_size, device=device)

    # Initialize Model
    model = ByteLatentTransformer(vocab_size=256, hidden_size=256, num_layers=4, num_heads=8, dropout_rate=0.1)
    
    # Train the model
    print("Starting Training...")
    model = train_model(
        model=model,
        dataloader=dataloader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device
    )
    print("Training Completed.")

    # Sampling Configuration
    sampler_config = SamplerConfig(
        low_entropy_threshold=0.3,
        medium_entropy_threshold=1.2,
        high_entropy_threshold=2.5
    )

    # Generate Text
    seed_text = "The quick brown fox jumps over the lazy dog."
    print("\nGenerating Text:")
    generated_text = generate_text(
        model=model,
        seed_text=seed_text,
        length=100,
        config=sampler_config,
        device=device
    )
    print(generated_text)

if __name__ == "__main__":
    main()
