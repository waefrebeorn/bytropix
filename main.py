# main.py

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, List
import math
from dataclasses import dataclass
from torch import amp
from torch.cuda.amp import autocast
import gc
import wandb
import argparse
import platform
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, is_initialized
import socket
import subprocess

# ----------------------------
# Dataset Preparation
# ----------------------------

class ByteDataset(Dataset):
    def __init__(self, data: np.memmap, context_size: int, stride: int = 1):
        """More efficient dataset implementation using memory-mapped data."""
        self.data = data
        self.context_size = context_size
        self.stride = stride
        self.length = (len(self.data) - context_size) // stride

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        context = self.data[start_idx:start_idx + self.context_size]
        target = self.data[start_idx + self.context_size]
        return torch.from_numpy(context).long(), torch.tensor(target).long()

# ----------------------------
# Sampler Configuration
# ----------------------------

@dataclass
class SamplerConfig:
    low_entropy_threshold: float = 0.3
    medium_entropy_threshold: float = 1.2
    high_entropy_threshold: float = 2.5

# ----------------------------
# Entropy-Based Sampling
# ----------------------------

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
    if low_mask.any():
        sampled[low_mask] = torch.argmax(probs[low_mask], dim=-1)

    # Medium entropy: top-k sampling
    med_mask = (entropy >= config.low_entropy_threshold) & (entropy < config.medium_entropy_threshold)
    if med_mask.any():
        top_k = 10
        top_k_probs, top_k_indices = torch.topk(probs[med_mask], k=top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
        sampled[med_mask] = top_k_indices.gather(1, sampled_indices.unsqueeze(-1)).squeeze(-1)

    # High entropy: random sampling
    high_mask = entropy >= config.medium_entropy_threshold
    if high_mask.any():
        sampled[high_mask] = torch.multinomial(probs[high_mask], num_samples=1).squeeze(-1)

    return sampled

# ----------------------------
# Model Architecture
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

class LocalEncoder(nn.Module):
    """Local encoder for processing byte sequences."""
    def __init__(self, hidden_size: int, num_heads: int = 4, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(256, hidden_size)
        self.transformer = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout_rate=0.1)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

        # Hash n-gram embeddings
        self.ngram_embeddings = nn.ModuleDict({
            f'ngram_{n}': nn.Embedding(500000, hidden_size)
            for n in range(3, 9)
        })

    def compute_ngram_embeddings(self, x):
        batch_size, seq_len = x.shape
        embeddings = []
        for n in range(3, 9):
            if seq_len >= n:
                # Simple rolling hash for n-grams
                ngrams = x.unfold(1, n, 1)  # Shape: (batch_size, seq_len - n +1, n)
                # Create a tensor for [256^0, 256^1, ..., 256^(n-1)]
                multiplier = torch.tensor([256**i for i in range(n)], device=x.device, dtype=torch.long)
                hashed = (ngrams * multiplier).sum(-1) % 500000  # Shape: (batch_size, seq_len -n +1)
                emb = self.ngram_embeddings[f'ngram_{n}'](hashed)  # Shape: (batch_size, seq_len -n +1, hidden_size)
                # Pad to match the original seq_len
                pad_size = n - 1
                emb = F.pad(emb, (0, 0, 0, pad_size), value=0)  # Shape: (batch_size, seq_len, hidden_size)
                embeddings.append(emb)
        return sum(embeddings) if embeddings else torch.zeros_like(self.embedding(x))

    def forward(self, x):
        x_emb = self.embedding(x)  # Shape: (batch_size, seq_len, hidden_size)
        ngram_emb = self.compute_ngram_embeddings(x)  # Shape: (batch_size, seq_len, hidden_size)
        x = x_emb + ngram_emb
        for layer in self.transformer:
            x = layer(x)
        return self.norm(x)  # Shape: (batch_size, seq_len, hidden_size)

class LocalDecoder(nn.Module):
    """Local decoder for generating bytes."""
    def __init__(self, hidden_size: int, num_heads: int = 4, num_layers: int = 9):
        super().__init__()
        self.transformer = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout_rate=0.1)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, 256)

    def forward(self, x):
        for layer in self.transformer:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)  # Shape: (batch_size, seq_len, 256)

class ByteLatentTransformer(nn.Module):
    """Improved Byte-Level Transformer architecture."""
    def __init__(self, hidden_size: int = 256, num_layers: int = 4, 
                 num_heads: int = 8, dropout_rate: float = 0.1, context_size: int = 8):
        super().__init__()
        self.context_size = context_size
        self.local_encoder = LocalEncoder(hidden_size)
        self.local_decoder = LocalDecoder(hidden_size)
        self.latent_transformer = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout_rate)

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
        # Local encoding
        local_features = self.local_encoder(x)  # Shape: (batch_size, seq_len, hidden_size)

        # Positional encoding
        local_features = self.pos_encoding(local_features)

        # Global processing
        latent = local_features
        for block in self.latent_transformer:
            latent = block(latent, attention_mask)
        latent = self.norm(latent)

        # Local decoding
        output = self.local_decoder(latent)  # Shape: (batch_size, seq_len, 256)
        return output

# ----------------------------
# Distributed Training Setup
# ----------------------------

def setup_distributed(args):
    """Initialize distributed training if multiple GPUs are available and enabled."""
    if args.distributed and torch.cuda.device_count() > 1:
        if platform.system() == "Windows":
            backend = "gloo"
        else:
            backend = "nccl"

        # Automatically find a free port for initialization
        port = find_free_port()
        init_method = f'tcp://127.0.0.1:{port}'

        world_size = torch.cuda.device_count()
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(port)

        try:
            init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=args.rank)
            torch.cuda.set_device(args.local_rank)
            device = torch.device(f'cuda:{args.local_rank}')
            return device, world_size
        except Exception as e:
            print(f"Distributed initialization failed: {e}")
            print("Falling back to single GPU training.")
            return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 1
    else:
        # Single GPU or CPU training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 1

def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# ----------------------------
# Data Loader Preparation
# ----------------------------

def prepare_dataloader(
    batch_size: int,
    context_size: int, 
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False
):
    """Optimized dataloader with memory mapping and distributed sampler."""
    # Memory map the data file instead of loading it entirely
    try:
        byte_data = np.memmap('data/wikitext_train.csv', dtype=np.uint8, mode='r')
    except FileNotFoundError:
        raise RuntimeError("Dataset file not found. Ensure 'data/wikitext_train.csv' exists.")
    except ValueError:
        raise RuntimeError("Dataset file has incompatible format or is corrupted.")

    dataset = ByteDataset(byte_data, context_size)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,  # Shuffle handled by sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2
    )

    return dataloader, sampler

# ----------------------------
# Training Utilities
# ----------------------------

def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    sampler: Optional[torch.utils.data.Sampler],
    epochs: int,
    learning_rate: float,
    device: torch.device,
    world_size: int,
    args,
    gradient_accumulation_steps: int = 4,  
):
    """Optimized training loop with mixed precision and distributed training."""
    
    # Initialize wandb only on the first process
    if args.rank == 0:
        wandb.init(project="blt-training", config={
            "epochs": epochs,
            "batch_size": dataloader.batch_size * world_size,
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "world_size": world_size
        })
    
    # Wrap model in DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None, find_unused_parameters=False)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs * len(dataloader)
    )

    # Initialize mixed precision training
    scaler = amp.GradScaler()
    
    model.train()
    for epoch in range(epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)  # More efficient than zeros
        
        for batch_idx, (context, target) in enumerate(dataloader):
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast():
                logits = model(context)
                loss = F.cross_entropy(logits[:, -1, :], target)
                loss = loss / gradient_accumulation_steps
            
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Log metrics only on the first process
                if args.rank == 0:
                    wandb.log({
                        "loss": loss.item() * gradient_accumulation_steps,
                        "lr": scheduler.get_last_lr()[0]
                    })
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Manual memory cleanup
            if batch_idx % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
            # Print progress only on the first process
            if args.rank == 0 and batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item() * gradient_accumulation_steps:.4f}')
        
        avg_loss = total_loss / len(dataloader)
        if args.rank == 0:
            print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')
            wandb.log({"epoch_loss": avg_loss})
        
        # Save checkpoint only on the first process
        if (epoch + 1) % 5 == 0 and args.rank == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch}.pt'
            if world_size > 1:
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')
    
    if args.rank == 0:
        wandb.finish()

# ----------------------------
# Text Generation
# ----------------------------

def generate_text(
    model: nn.Module,
    seed_text: str,
    length: int,
    config: SamplerConfig,
    device: torch.device = torch.device("cuda"),
    temperature: float = 0.8
):
    """Generate text using the trained model."""
    model.eval()
    generated = list(seed_text.encode('utf-8'))

    with torch.no_grad():
        for _ in range(length):
            # Ensure we have enough context
            if len(generated) < model.module.context_size:
                context_bytes = generated
                # Pad if necessary
                context_bytes = [0]*(model.module.context_size - len(context_bytes)) + context_bytes
            else:
                context_bytes = generated[-model.module.context_size:]

            context = torch.tensor([context_bytes], dtype=torch.long).to(device)

            with autocast():
                logits = model(context)
                last_logits = logits[:, -1, :] / temperature  # Apply temperature

            # Sample next token based on entropy
            sampled_token = entropy_based_sampling(last_logits, config)
            generated.append(int(sampled_token))

    return bytes(generated).decode('utf-8', errors='replace')

# ----------------------------
# Argument Parsing
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Byte-Level Transformer Training Script")
    parser.add_argument('--distributed', action='store_true', help='Enable Distributed Data Parallel training')
    parser.add_argument('--rank', type=int, default=0, help='Rank of the current process')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    return parser.parse_args()

# ----------------------------
# Main Function
# ----------------------------

def main():
    args = parse_args()
    
    # Initialize distributed training
    device, world_size = setup_distributed(args)
    
    # Configuration
    config = {
        "context_size": 8,
        "batch_size": 32,  # Reduced batch size per GPU
        "epochs": 20,
        "learning_rate": 1e-3,
        "num_workers": 4
    }
    
    # Initialize model
    model = ByteLatentTransformer(
        hidden_size=256,
        num_layers=4, 
        num_heads=8,
        dropout_rate=0.1,
        context_size=config["context_size"]
    ).to(device)
    
    # Prepare DataLoader
    dataloader, sampler = prepare_dataloader(
        batch_size=config["batch_size"],
        context_size=config["context_size"],
        num_workers=config["num_workers"],
        distributed=(world_size > 1)
    )
    
    # Train with optimizations
    train_model(
        model=model,
        dataloader=dataloader,
        sampler=sampler,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        device=device,
        world_size=world_size,
        args=args
    )
    
    # Sampling Configuration
    sampler_config = SamplerConfig(
        low_entropy_threshold=0.3,
        medium_entropy_threshold=1.2,
        high_entropy_threshold=2.5
    )
    
    # Generate Text (Ensure this runs only on one process to avoid multiple outputs)
    if args.rank == 0:
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
    
    # Clean up distributed training
    if is_initialized():
        destroy_process_group()

if __name__ == "__main__":
    main()
