import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, List, Dict, Any, Iterable
import math
from dataclasses import dataclass
from torch import amp
from torch.cuda.amp import autocast  # Updated autocast usage
import gc
import wandb
import argparse
import platform
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, is_initialized
import socket
from collections import deque
import logging

from EnhancedSGD import EnhancedSGD  # Ensure this is correctly implemented and accessible

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ----------------------------
# Dataset Preparation
# ----------------------------

class WikiTextByteDataset(Dataset):
    """Dataset for byte-level training on WikiText data."""
    def __init__(self, csv_path: str, context_size: int, column_name: str = 'text'):
        """
        Args:
            csv_path (str): Path to the CSV file containing text data.
            context_size (int): Size of the context window.
            column_name (str): Name of the column containing the text data.
        """
        super().__init__()
        self.context_size = context_size

        # Load CSV data
        logging.info(f"Loading data from {csv_path}")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logging.error(f"Failed to read CSV file at {csv_path}: {e}")
            raise e

        if column_name not in df.columns:
            logging.error(f"Column '{column_name}' not found in CSV file.")
            raise ValueError(f"Column '{column_name}' not found in CSV file.")

        # Concatenate all text entries with newlines
        full_text = '\n'.join(df[column_name].fillna('').astype(str))

        # Convert text to bytes
        self.bytes_data = torch.tensor([byte for byte in full_text.encode('utf-8')], dtype=torch.long)
        logging.info(f"Total bytes in dataset: {len(self.bytes_data)}")

    def __len__(self):
        # Length is total bytes minus context size (need room for target)
        return max(0, len(self.bytes_data) - self.context_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get context window and target byte
        context = self.bytes_data[idx:idx + self.context_size]
        target = self.bytes_data[idx + self.context_size]

        # Ensure types are correct
        return context.long(), target.long()

def prepare_dataloaders(
    train_csv: str,
    test_csv: str,
    batch_size: int,
    context_size: int,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare train and test DataLoaders from CSV files.

    Args:
        train_csv (str): Path to training CSV file.
        test_csv (str): Path to test CSV file.
        batch_size (int): Batch size.
        context_size (int): Context window size.
        num_workers (int): Number of worker processes.
        pin_memory (bool): Whether to pin memory for GPU transfer.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = WikiTextByteDataset(
        csv_path=train_csv,
        context_size=context_size
    )

    test_dataset = WikiTextByteDataset(
        csv_path=test_csv,
        context_size=context_size
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    return train_loader, test_loader

# Utility function to decode bytes back to text
def decode_bytes(byte_tensor: torch.Tensor) -> str:
    """Convert a tensor of byte values back to text."""
    return bytes(byte_tensor.cpu().numpy().tolist()).decode('utf-8', errors='replace')

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
# Custom Loss Function
# ----------------------------

def cross_entropy_with_temperature(logits: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute cross-entropy loss with temperature scaling for numerical stability.

    Args:
        logits (torch.Tensor): Logits from the model of shape (batch_size, num_classes).
        targets (torch.Tensor): Target indices of shape (batch_size,).
        temperature (float): Temperature scaling factor.

    Returns:
        torch.Tensor: Computed loss.
    """
    scaled_logits = logits / temperature
    return F.cross_entropy(scaled_logits, targets)

# ----------------------------
# Adaptive Spline Activation
# ----------------------------

class AdaptiveSplineActivation(nn.Module):
    def __init__(self, hidden_size, num_knots=10, entropy_scaling=True, eps=1e-6):
        super().__init__()
        # Initialize knots in a more stable range
        self.knots = nn.Parameter(torch.linspace(-0.5, 0.5, num_knots))
        # Initialize weights closer to 1 for better gradient flow
        self.weights = nn.Parameter(torch.ones(num_knots) + torch.randn(num_knots) * 0.01)
        self.entropy_scaling = entropy_scaling
        self.eps = eps
        self.hidden_size = hidden_size

        if entropy_scaling:
            self.entropy_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 16),
                nn.LayerNorm(hidden_size // 16, eps=self.eps),
                nn.ReLU(),
                nn.Linear(hidden_size // 16, 1),
                nn.Sigmoid()  # Added sigmoid to ensure positive scaling
            )
            # Initialize entropy projection with smaller weights
            for layer in self.entropy_proj.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Add LayerNorm before normalization to stabilize
        layer_norm = nn.functional.layer_norm(x, x.shape[-1:])
        # Use softer normalization
        norm = torch.sqrt(torch.sum(layer_norm * layer_norm, dim=-1, keepdim=True) + self.eps)
        x_normalized = layer_norm / (norm + 1.0)  # Add 1.0 to prevent division by small numbers

        # Compute distances with stable scaling
        distances = torch.abs(x_normalized.unsqueeze(-1) - self.knots)
        max_dist = distances.max(dim=-1, keepdim=True)[0] + self.eps
        distances = distances / max_dist

        # Softmax with temperature for more stable weights
        weights = torch.softmax(-distances * 5.0, dim=-1)

        if self.entropy_scaling:
            entropy = self.entropy_proj(x)
            # Limit the scaling factor range
            scale_factor = 1.0 + 0.1 * entropy
            weights = weights * scale_factor.unsqueeze(-1)

        # Compute output with gradient clipping
        output = (weights * self.weights).sum(-1)
        return torch.clamp(output, -1.0, 1.0)

# ----------------------------
# Entropy-Guided Attention
# ----------------------------

class EntropyGuidedAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Single projection matrices for Q, K, V
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.zeros_(self.qkv.bias)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1.0e-6)
        
        # Entropy projection with proper initialization
        self.entropy_proj = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        nn.init.normal_(self.entropy_proj[0].weight, std=0.02)
        nn.init.zeros_(self.entropy_proj[0].bias)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)
        
        self.dropout = nn.Dropout(dropout_rate)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Create causal mask that matches the expected shape
        # Shape: [seq_len, seq_len]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.bool()  # Convert to boolean mask
        return ~mask  # Invert to get the causal mask

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        device = x.device

        # Apply layer norm first
        x = self.layer_norm(x)

        # Project all Q, K, V at once and split
        qkv = self.qkv(x)  # Shape: [batch_size, seq_len, 3 * hidden_size]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape [batch_size, num_heads, seq_len, head_dim]

        # Compute attention scores
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, device)
        # Expand mask for batch size and heads
        # Shape: [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Apply causal mask
        attention_scores = attention_scores.masked_fill(~causal_mask, float('-inf'))

        # Compute entropy-based scaling
        # [batch_size, seq_len, 1]
        entropy_scale = self.entropy_proj(x)
        # Reshape for broadcasting: [batch_size, 1, seq_len, 1]
        entropy_scale = entropy_scale.permute(0, 2, 1).unsqueeze(1)
        entropy_scale = torch.clamp(entropy_scale, min=0.1, max=2.0)

        # Apply attention weights
        attention_probs = torch.softmax(attention_scores / 0.1, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply entropy scaling to attention probabilities
        attention_probs = attention_probs * entropy_scale

        # Compute attention output
        # [batch_size, num_heads, seq_len, head_dim]
        attention_output = torch.matmul(attention_probs, v)

        # Reshape attention output
        # [batch_size, seq_len, num_heads, head_dim]
        attention_output = attention_output.transpose(1, 2).contiguous()
        # [batch_size, seq_len, hidden_size]
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_size)

        # Final projection and scaling
        output = self.out_proj(attention_output)
        output = output * 0.1  # Scale down output

        return output

# ----------------------------
# Positional Encoding
# ----------------------------

class PositionalEncoding(nn.Module):
    """Enhanced positional encoding with dropout and normalization."""
    def __init__(self, hidden_size: int, dropout_rate: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(1, max_len, hidden_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.norm(x)
        return self.dropout(x)

# ----------------------------
# Enhanced Transformer Block
# ----------------------------

class EnhancedTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1, use_adaptive_activations=True):
        super().__init__()
        self.attention = EntropyGuidedAttention(hidden_size, num_heads, dropout_rate)

        # Modified feed-forward network configuration
        ff_dim = 4 * hidden_size
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            AdaptiveSplineActivation(ff_dim) if use_adaptive_activations else nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout_rate)
        )

        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x, attention_mask=None):
        attn_output = self.attention(x, attention_mask)
        # Log tensor shapes before residual connection
        logging.debug(f"Layer {id(self)} - Before residual add (Attention): x.shape={x.shape}, attn_output.shape={attn_output.shape}")
        if x.shape != attn_output.shape:
            logging.error(f"Shape mismatch in attention: x.shape={x.shape}, attn_output.shape={attn_output.shape}")
            raise ValueError("Shape mismatch in attention output")
        x = self.norm1(x + attn_output)

        ff_output = self.feed_forward(x)
        # Log tensor shapes before residual connection
        logging.debug(f"Layer {id(self)} - Before residual add (FeedForward): x.shape={x.shape}, ff_output.shape={ff_output.shape}")
        if x.shape != ff_output.shape:
            logging.error(f"Shape mismatch in feed_forward: x.shape={x.shape}, ff_output.shape={ff_output.shape}")
            raise ValueError("Shape mismatch in feed_forward output")
        return self.norm2(x + ff_output)

# ----------------------------
# Gradient Management
# ----------------------------

class GradientManager:
    """Manages gradient statistics and adaptations during training."""
    def __init__(self, model: nn.Module, window_size: int = 100):
        self.model = model
        self.grad_history = deque(maxlen=window_size)
        self.entropy_history = deque(maxlen=window_size)
        self.param_groups = self._group_parameters()

    def _group_parameters(self) -> Dict[str, list]:
        """Group parameters by layer type for targeted optimization."""
        groups = {
            'attention': [],
            'feedforward': [],
            'embedding': [],
            'norm': []
        }

        for name, param in self.model.named_parameters():
            if 'attention' in name:
                groups['attention'].append(param)
            elif 'feed_forward' in name or 'feedforward' in name:
                groups['feedforward'].append(param)
            elif 'embedding' in name:
                groups['embedding'].append(param)
            elif 'norm' in name:
                groups['norm'].append(param)

        return groups

    def compute_grad_stats(self) -> Dict[str, float]:
        """Compute gradient statistics for adaptation."""
        stats = {}
        for group_name, params in self.param_groups.items():
            group_grads = [p.grad for p in params if p.grad is not None]
            if group_grads:
                grad_tensor = torch.cat([g.flatten() for g in group_grads])
                stats[f'{group_name}_norm'] = torch.norm(grad_tensor).item()
                stats[f'{group_name}_var'] = torch.var(grad_tensor).item()

        return stats

    def update_histories(self, loss: float, grad_stats: Dict[str, float]):
        """Update gradient and loss histories."""
        self.grad_history.append(grad_stats)
        entropy = float(-loss * math.log(loss + 1e-10) if loss > 0 else 0)
        self.entropy_history.append(entropy)

# ----------------------------
# Enhanced Byte-Level Transformer
# ----------------------------

class EnhancedByteTransformer(nn.Module):
    def __init__(
        self, 
        hidden_size: int = 512,            # Increased for byte representation
        num_heads: int = 8,                # Correctly assigned
        num_layers: int = 6,               # Increased depth for byte patterns
        dropout_rate: float = 0.1,         # Reduced dropout for bytes
        context_size: int = 32,            # Correctly assigned
        use_adaptive_activations: bool = True
    ):
        super().__init__()
        self.context_size = context_size  # Sequence length

        # Stabilize embedding initialization
        self.byte_embeddings = nn.Embedding(256, hidden_size)
        nn.init.normal_(self.byte_embeddings.weight, mean=0.0, std=0.02)  # Increased std slightly

        self.pos_encoding = PositionalEncoding(hidden_size, dropout_rate)

        # Add input normalization
        self.input_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # Transformer layers
        self.layers = nn.ModuleList([
            EnhancedTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                use_adaptive_activations=use_adaptive_activations
            ) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # Modified output layer with careful initialization
        self.output = nn.Linear(hidden_size, 256)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)

        # Initialize gradient manager
        self.grad_manager = GradientManager(self)

    def forward(self, x):
        # Get embeddings
        x = self.byte_embeddings(x)  # [batch, seq_len, hidden_size]

        # Apply positional encoding
        x = self.pos_encoding(x)     # [batch, seq_len, hidden_size]

        # Initial normalization
        x = self.input_norm(x)       # [batch, seq_len, hidden_size]

        # Scale embeddings to prevent explosion
        x = x * math.sqrt(x.size(-1))  # [batch, seq_len, hidden_size]

        # Create causal attention mask
        attention_mask = torch.tril(torch.ones(x.size(1), x.size(1), device=x.device)).bool()

        # Process through transformer layers with additional checks
        for idx, layer in enumerate(self.layers):
            try:
                x = layer(x, attention_mask)
            except ValueError as e:
                logging.error(f"Error in layer {idx}: {e}")
                raise e

            # Prevent explosion
            if torch.isnan(x).any() or torch.isinf(x).any():
                logging.warning(f"Layer {idx} output contains NaN or Inf. Reverting to residual connection.")
                x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)

            x = torch.clamp(x, min=-100.0, max=100.0)  # Add reasonable bounds

        x = self.final_norm(x)  # [batch, seq_len, hidden_size]

        # Add pre-output normalization
        logits = self.output(x)  # [batch, seq_len, 256]

        # Scale logits to prevent softmax instability
        logits = logits / math.sqrt(x.size(-1))  # [batch, seq_len, 256]

        return logits

    def get_optimization_groups(self) -> List[Dict[str, Any]]:
        """Define parameter groups for optimized training."""
        return [
            {
                'params': self.grad_manager.param_groups['attention'],
                'lr_mult': 1.0,   # Full learning rate for attention
                'weight_decay': 0.01,
                'type': 'attention'
            },
            {
                'params': self.grad_manager.param_groups['feedforward'],
                'lr_mult': 0.8,   # Slightly reduced for feedforward
                'weight_decay': 0.01,
                'type': 'feedforward'
            },
            {
                'params': self.grad_manager.param_groups['embedding'],
                'lr_mult': 0.5,   # Reduced for byte embeddings
                'weight_decay': 0.0,
                'type': 'embedding'
            },
            {
                'params': self.grad_manager.param_groups['norm'],
                'lr_mult': 0.01,  # Further reduced for normalization
                'weight_decay': 0.0,
                'type': 'norm'
            }
        ]

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
            logging.info(f"Distributed training initialized on device {device}")
            return device, world_size
        except Exception as e:
            logging.error(f"Distributed initialization failed: {e}")
            logging.info("Falling back to single GPU training.")
            return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 1
    else:
        # Single GPU or CPU training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Training on device {device}")
        return device, 1

def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# ----------------------------
# Validation Utilities
# ----------------------------

def compute_validation_loss(dataloader: DataLoader, model: nn.Module, device: torch.device) -> float:
    """Compute average loss on the validation dataset."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for context, target in dataloader:
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda'):
                logits = model(context)
                # Clamp logits to prevent NaNs
                logits = torch.clamp(logits, min=-10.0, max=10.0)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logging.warning("Invalid logits detected in validation. Skipping this batch.")
                    continue
                if target.min() < 0 or target.max() >= logits.size(-1):
                    logging.warning(f"Invalid target values in validation: {target}")
                    continue
                loss = cross_entropy_with_temperature(logits[:, -1, :], target)
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning("NaN detected in validation loss. Skipping this batch.")
                    continue
                total_loss += loss.item()
    model.train()
    return total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')

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
    gradient_accumulation_steps: int = 4,  # Adjusted for better performance
    validation_dataloader: Optional[DataLoader] = None
):
    """Optimized training loop with fixed gradient scaling and accumulation."""
    
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
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)

    # Get optimization groups
    optimization_groups = (
        model.module.get_optimization_groups() if world_size > 1 else model.get_optimization_groups()
    )

    optimizer = EnhancedSGD(
        optimization_groups,
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.01,
        smoothing_factor=0.03,
        entropy_threshold=0.2,
        max_grad_norm=0.5,
        noise_scale=1e-5,
        lr_scale_bounds=(0.8, 1.2),
        momentum_scale_bounds=(0.8, 1.2)
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(dataloader) // gradient_accumulation_steps,
        pct_start=0.1,
        div_factor=8.0,
        final_div_factor=50,
        anneal_strategy='cos'
    )

    scaler = amp.GradScaler(
        init_scale=2**10,
        growth_factor=1.5,
        backoff_factor=0.5,
        growth_interval=100
    )

    model.train()

    for epoch in range(epochs):
        if world_size > 1 and sampler is not None:
            sampler.set_epoch(epoch)

        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (context, target) in enumerate(dataloader):
            # Validate input data
            if context.ndim != 2 or target.ndim != 1:
                logging.error(f"Invalid input shape: context={context.shape}, target={target.shape}. Skipping batch.")
                continue

            if context.size(0) != target.size(0):
                logging.error(f"Mismatch in batch size: context={context.size(0)}, target={target.size(0)}. Skipping batch.")
                continue

            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            try:
                # Forward pass with gradient scaling
                with torch.amp.autocast('cuda'):
                    logits = model(context)
                    logits = torch.clamp(logits, min=-10.0, max=10.0)

                    loss = cross_entropy_with_temperature(
                        logits[:, -1, :],
                        target,
                        temperature=1.0
                    ) / gradient_accumulation_steps

                # Backward pass
                scaler.scale(loss).backward()

                # Gradient accumulation step
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Unscale gradients
                    scaler.unscale_(optimizer)

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item() * gradient_accumulation_steps

                # Log metrics
                if args.rank == 0 and batch_idx % 100 == 0:
                    _log_training_metrics(
                        epoch, batch_idx, loss.item(),
                        scheduler.get_last_lr()[0]
                    )

            except RuntimeError as e:
                logging.error(f"Runtime error at Epoch {epoch}, Batch {batch_idx}: {e}")
                continue

        # Perform memory cleanup after epoch
        gc.collect()
        torch.cuda.empty_cache()

        # End of epoch processing
        _process_epoch_end(
            epoch, total_loss, len(dataloader),
            model, optimizer, scheduler, scaler,
            validation_dataloader, device, args
        )

    # Ensure model is saved at the end of training
    _save_checkpoint(
        epoch=epochs,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler
    )

def _validate_batch(context: torch.Tensor, target: torch.Tensor) -> bool:
    """Validate input tensors."""
    if torch.isnan(context).any() or torch.isnan(target).any():
        logging.warning("NaN detected in input tensors. Skipping batch.")
        return False
    if (target < 0).any() or (target >= 256).any():
        logging.warning("Target values out of range. Skipping batch.")
        return False
    return True

def _log_training_metrics(epoch: int, batch_idx: int, loss: float, lr: float):
    """Log training metrics to wandb and console."""
    wandb.log({
        "loss": loss,
        "learning_rate": lr
    })
    logging.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}, LR: {lr:.6f}')

def _process_epoch_end(
    epoch: int,
    total_loss: float,
    num_batches: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    scaler: amp.GradScaler,
    validation_dataloader: Optional[DataLoader],
    device: torch.device,
    args: Any
):
    """Process end of epoch tasks including validation and checkpointing."""
    if args.rank == 0:
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        logging.info(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')
        wandb.log({"epoch_loss": avg_loss})

        if validation_dataloader and (epoch + 1) % 2 == 0:
            val_loss = compute_validation_loss(
                validation_dataloader, model, device
            )
            logging.info(f'Validation Loss after Epoch {epoch}: {val_loss:.4f}')
            wandb.log({"val_loss": val_loss})

        if (epoch + 1) % 5 == 0:
            _save_checkpoint(
                epoch, model, optimizer, scheduler, scaler
            )

def _save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    scaler: amp.GradScaler
):
    """Save model checkpoint."""
    checkpoint_path = f'checkpoint_epoch_{epoch}.pt'
    model_state_dict = model.module.state_dict() if isinstance(
        model, DDP
    ) else model.state_dict()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }, checkpoint_path)
    logging.info(f'Checkpoint saved at {checkpoint_path}')

# ----------------------------
# Sampling Function
# ----------------------------

def generate_text(model: nn.Module, seed_text: str, length: int, config: SamplerConfig, device: torch.device) -> str:
    """
    Generate text using the trained model based on the seed_text.

    Args:
        model (nn.Module): Trained model.
        seed_text (str): Seed text to start generation.
        length (int): Number of bytes to generate.
        config (SamplerConfig): Configuration for entropy-based sampling.
        device (torch.device): Device to run the model on.

    Returns:
        str: Generated text.
    """
    model.eval()
    generated = seed_text.encode('utf-8')
    context_size = model.context_size
    context = torch.tensor([byte for byte in generated[-context_size:]], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(length):
            logits = model(context)
            next_byte_logits = logits[:, -1, :]  # Get logits for the last position
            sampled_byte = entropy_based_sampling(next_byte_logits, config)
            generated += sampled_byte.cpu().numpy().tolist()
            # Update context
            context = torch.cat([context[:, 1:], sampled_byte.unsqueeze(0)], dim=1)

    return decode_bytes(torch.tensor(generated))

# ----------------------------
# Argument Parsing
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Byte-Level Transformer Training Script")
    parser.add_argument('--distributed', action='store_true', help='Enable Distributed Data Parallel training')
    parser.add_argument('--rank', type=int, default=0, help='Rank of the current process')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU')  # Increased to 128
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Initial learning rate')  # Increased to 3e-4
    parser.add_argument('--context_size', type=int, default=32, help='Context size for the model')  # Correctly assigned
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--validation', action='store_true', help='Enable validation during training')
    parser.add_argument('--validation_batch_size', type=int, default=64, help='Batch size for validation')
    parser.add_argument('--no_anomaly', action='store_true', help='Disable anomaly detection for faster training')
    return parser.parse_args()

# ----------------------------
# Main Function
# ----------------------------

def main():
    args = parse_args()
    device, world_size = setup_distributed(args)

    # Enhanced configuration for byte-level training
    config = {
        "context_size": args.context_size,         # Correctly assigned from arguments
        "batch_size": args.batch_size,             # Correctly assigned from arguments
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "warmup_steps": 2000,
        "gradient_clip_val": 0.5,                  # Reduced for byte stability
        "weight_decay": 0.01
    }

    # Prepare DataLoaders BEFORE initializing the scheduler
    train_loader, test_loader = prepare_dataloaders(
        train_csv="data/wikitext_train.csv",  # Update dataset file path as needed
        test_csv="data/wikitext_test.csv",    # Update dataset file path as needed
        batch_size=config["batch_size"],
        context_size=config["context_size"],
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # Initialize model with optimized byte architecture
    model = EnhancedByteTransformer(
        hidden_size=512,            # Increased for byte representation
        num_heads=8,                # Correctly assigned
        num_layers=6,               # Increased depth for byte patterns
        dropout_rate=0.1,           # Reduced dropout for bytes
        context_size=config["context_size"],  # Correctly assigned
        use_adaptive_activations=True
    ).to(device)

    # Get optimization groups with byte-specific learning rates
    optimization_groups = model.get_optimization_groups()

    # Adjust learning rates and momentums based on lr_mult and momentum_scale
    for group in optimization_groups:
        if group['type'] == 'attention':
            group['lr_mult'] = 1.0   # Full learning rate for attention
        elif group['type'] == 'feedforward':
            group['lr_mult'] = 0.8   # Slightly reduced for feedforward
        elif group['type'] == 'embedding':  
            group['lr_mult'] = 0.5   # Reduced for byte embeddings
        elif group['type'] == 'norm':
            group['lr_mult'] = 0.01  # Further reduced for normalization

    # Initialize optimizer with byte-specific parameters
    optimizer = EnhancedSGD(
        optimization_groups,
        lr=config["learning_rate"],
        momentum=0.9,
        weight_decay=config["weight_decay"],
        smoothing_factor=0.03,      # Reduced for byte stability
        entropy_threshold=0.2,      # Lower threshold for bytes
        max_grad_norm=config["gradient_clip_val"],
        noise_scale=1e-5,
        lr_scale_bounds=(0.8, 1.2),
        momentum_scale_bounds=(0.8, 1.2)
    )

    # ByteOptimal learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        epochs=config["epochs"],
        steps_per_epoch=len(train_loader) // gradient_accumulation_steps_from_args(),
        pct_start=0.1,
        div_factor=8.0,            # Reduced for byte learning
        final_div_factor=50,       # Adjusted for bytes
        anneal_strategy='cos'
    )

    # Initialize scaler
    scaler = amp.GradScaler(
        init_scale=2**10,  # Start with a smaller scale
        growth_factor=1.5,  # More conservative growth
        backoff_factor=0.5,  # More aggressive backoff
        growth_interval=100  # Increase scale less frequently
    )

    model.train()

    # Inspect a few samples from the dataset
    def inspect_dataset(dataset, num_samples=5):
        logging.info("Inspecting dataset samples:")
        for i in range(num_samples):
            context, target = dataset[i]
            logging.info(f"\nSample {i}:")
            logging.info(f"Context bytes: {context.tolist()}")
            logging.info(f"Target byte: {target.item()}")
            logging.info(f"Decoded context: {decode_bytes(context)}")
            logging.info(f"Decoded target: {bytes([target.item()]).decode('utf-8', errors='replace')}")

    inspect_dataset(train_loader.dataset, num_samples=5)

    # Optionally prepare validation DataLoader
    validation_dataloader = None

    if args.validation:
        # Path to the validation dataset
        validation_data_path = "data/wikitext_validation.csv"  # Update as needed

        try:
            validation_dataset = WikiTextByteDataset(
                csv_path=validation_data_path,
                context_size=config["context_size"]
            )
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=args.validation_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            logging.info(f"Validation dataset size: {len(validation_dataset)}")
            inspect_dataset(validation_dataloader.dataset, num_samples=5)

        except FileNotFoundError:
            logging.error(f"Validation dataset file '{validation_data_path}' not found. Skipping validation.")
        except ValueError as e:
            logging.error(f"Validation dataset file at '{validation_data_path}' has an incompatible format or is corrupted. Skipping validation.")
            logging.error(f"Error details: {e}")

    # Train with byte-optimized parameters
    train_model(
        model=model,
        dataloader=train_loader,
        sampler=None,  # Assuming non-distributed; adjust if needed
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        device=device,
        world_size=world_size,
        args=args,
        gradient_accumulation_steps=gradient_accumulation_steps_from_args(),
        validation_dataloader=validation_dataloader
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
        logging.info("\nGenerating Text:")
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

def gradient_accumulation_steps_from_args() -> int:
    """Determine gradient accumulation steps based on arguments or defaults."""
    # You can modify this function to derive steps from other arguments if needed
    return 4  # Default value; adjust as needed

if __name__ == "__main__":
    main()
