# =================================================================================================
#
#        GPT-at-Home & wubumind: The Phoenix Protocol (v6.1 - On-the-Fly Hashing)
#
#       This version adapts the data pipeline to handle extremely large datasets.
#       Instead of pre-calculating and saving all hashes (which caused a memory error),
#       hashes are now computed on-the-fly for each batch during training. This makes the
#       pre-tokenization step memory-safe and allows the framework to scale to any dataset size.
#
# =================================================================================================

import os
import sys
import argparse
import pickle
import time
from pathlib import Path

# --- Environment Setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Keep TensorFlow quiet

# --- Core PyTorch Imports for the New Engine ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np
from tqdm import tqdm
from numpy.lib.stride_tricks import as_strided

try:
    from rich.console import Console
    from rich.markdown import Markdown
except ImportError:
    print("Rich library not found, chat formatting will be basic. `pip install rich`")
    Console, Markdown = None, None

# ==============================================================================
# 1. ENGINE: WuBu Memory WBA (PyTorch)
# ==============================================================================


class WuBuSparseAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, k: int, working_memory_size: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads, self.d_head, self.k, self.working_memory_size = d_model, num_heads, d_model // num_heads, k, working_memory_size
        self.w_q, self.w_k, self.w_v, self.w_o = (nn.Linear(d_model, d_model) for _ in range(4))
        indexer_dim = 64
        self.w_q_indexer, self.w_k_indexer = nn.Linear(d_model, indexer_dim), nn.Linear(d_model, indexer_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, _ = x.shape
        q, k, v = [proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2) for proj in (self.w_q, self.w_k, self.w_v)]
        
        if seq_len <= self.working_memory_size:
            all_attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
            all_v = v
        else:
            k_work, v_work = k[:, :, -self.working_memory_size:, :], v[:, :, -self.working_memory_size:, :]
            k_assoc, v_assoc = k[:, :, :-self.working_memory_size, :], v[:, :, :-self.working_memory_size:, :]
            
            attn_scores_work = torch.matmul(q, k_work.transpose(-2, -1)) / math.sqrt(self.d_head)
            
            assoc_len = k_assoc.shape[2]
            q_idx, k_idx = self.w_q_indexer(x), self.w_k_indexer(x[:, :assoc_len, :])
            indexer_scores = F.relu(torch.matmul(q_idx, k_idx.transpose(-1, -2)))
            _, top_k_indices = torch.topk(indexer_scores, min(self.k, assoc_len), dim=-1)
            
            indices_for_gather = top_k_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, -1, self.d_head)
            k_assoc_expanded = k_assoc.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
            k_assoc_gathered = torch.gather(k_assoc_expanded, 3, indices_for_gather)
            v_assoc_expanded = v_assoc.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
            v_assoc_gathered = torch.gather(v_assoc_expanded, 3, indices_for_gather)
            
            q_expanded = q.unsqueeze(3)
            attn_scores_assoc = torch.matmul(q_expanded, k_assoc_gathered.transpose(-2, -1)).squeeze(3) / math.sqrt(self.d_head)
            all_attn_scores = torch.cat([attn_scores_assoc, attn_scores_work], dim=-1)
            
            v_work_expanded = v_work.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
            all_v = torch.cat([v_assoc_gathered, v_work_expanded], dim=3)
        
        # The internal masking logic I provided before is still better than passing the mask.
        if seq_len <= self.working_memory_size:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            all_attn_scores = all_attn_scores.masked_fill(causal_mask, -float('inf'))
        else:
            work_len = self.working_memory_size
            k_sparse = all_attn_scores.shape[-1] - work_len
            work_mask = torch.triu(torch.ones(seq_len, work_len, device=x.device), diagonal=1 + (seq_len - work_len)).bool()
            all_attn_scores[:, :, :, -work_len:].masked_fill_(work_mask, -float('inf'))

        attn_weights = F.softmax(all_attn_scores, dim=-1)

        if len(all_v.shape) == 5:
             output = torch.matmul(attn_weights.unsqueeze(3), all_v).squeeze(3)
        else:
             output = torch.matmul(attn_weights, all_v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(output)
        
        
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, k: int, working_memory_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = WuBuSparseAttention(d_model, num_heads, k, working_memory_size)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None): # Mask is now handled internally by attention
        x = self.norm1(x + self.dropout(self.attention(x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): return x + self.pe[:, :x.size(1)]

class WuBuMemoryWBA_Model(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, k, working_memory_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.hash_projector = nn.Linear(1, d_model, bias=False) 
        self.bridge_proj = nn.Linear(d_model, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, k, working_memory_size, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, indices, hashes):
        token_embed = self.token_embedding(indices)
        hash_embed = self.hash_projector(hashes.unsqueeze(-1).float())
        x = self.bridge_proj(token_embed + hash_embed) * math.sqrt(self.d_model)
        
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x) # Mask is no longer passed
        return self.output_layer(x)

# ==============================================================================
# 2. ROBUST DATA & TRAINING FRAMEWORK (ADAPTED FOR TORCH & ON-THE-FLY HASHING)
# ==============================================================================

class RollingHasher:
    def __init__(self, window_size, base=31, modulus=10**9 + 7):
        self.window_size, self.base, self.modulus = window_size, np.int64(base), np.int64(modulus)
        self.base_powers = np.power(self.base, np.arange(self.window_size - 1, -1, -1), dtype=np.int64)

    def hash_batch(self, batch_windows: np.ndarray) -> np.ndarray:
        """EFFICIENT METHOD: Calculates hashes for a batch of pre-made windows."""
        return (np.einsum('ijk,k->ij', batch_windows.astype(np.int64), self.base_powers, optimize=True) % self.modulus).astype(np.int32)

def data_generator(indices_mmap, seed, per_device_batch_size, block_size, hash_window, num_devices):
    hasher = RollingHasher(window_size=hash_window)
    total_batch_size = per_device_batch_size * num_devices
    
    first_valid_start_idx = hash_window - 1
    num_examples = len(indices_mmap) - block_size - first_valid_start_idx
    
    rng = np.random.default_rng(seed)
    CHUNK_SIZE = 1_000_000
    while True:
        chunk_starts = np.arange(first_valid_start_idx, num_examples, CHUNK_SIZE); rng.shuffle(chunk_starts)
        for start in chunk_starts:
            end = min(start + CHUNK_SIZE, num_examples)
            chunk_indices = np.arange(start, end)[rng.permutation(end - start)]
            
            for i in range(0, len(chunk_indices), total_batch_size):
                batch_start_indices = chunk_indices[i : i + total_batch_size]
                if len(batch_start_indices) < total_batch_size: continue
                
                batch_y_indices = np.stack([indices_mmap[s + 1 : s + block_size + 1] for s in batch_start_indices])
                batch_x_indices = np.stack([indices_mmap[s : s + block_size] for s in batch_start_indices])

                full_sequences = np.stack([indices_mmap[s - first_valid_start_idx : s + block_size] for s in batch_start_indices])
                
                shape = (full_sequences.shape[0], block_size, hash_window)
                strides = (full_sequences.strides[0], full_sequences.strides[1], full_sequences.strides[1])
                hash_windows = as_strided(full_sequences, shape=shape, strides=strides)

                batch_x_hashes = hasher.hash_batch(hash_windows)

                yield {
                    'indices': torch.from_numpy(batch_x_indices),
                    'hashes': torch.from_numpy(batch_x_hashes),
                    'targets': torch.from_numpy(batch_y_indices)
                }

def predict_step_torch(model, indices, hashes, temperature=0.7, top_p=0.95):
    model.eval()
    with torch.no_grad():
        logits = model(indices, hashes)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits.scatter_(1, indices_to_remove, -float('Inf'))
        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        return next_token.item()



def train_step_torch(model, optimizer, criterion, batch, device):
    model.train()
    indices = batch['indices'].to(device, dtype=torch.long)
    hashes = batch['hashes'].to(device)
    targets = batch['targets'].to(device, dtype=torch.long)
    
    optimizer.zero_grad()
    logits = model(indices, hashes)
    
    # --- FIX IS HERE ---
    # Change '--1' to '-1' to correctly flatten the targets tensor.
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()



def save_checkpoint_torch(model, optimizer, epoch, filename):
    save_obj = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    with open(filename, 'wb') as f: pickle.dump(save_obj, f)
    print(f"\n--- PyTorch Checkpoint saved at epoch {epoch} to {filename} ---")

def load_checkpoint_torch(model, optimizer, filename, device):
    if not os.path.exists(filename): return 0
    with open(filename, 'rb') as f: save_obj = pickle.load(f)
    model.load_state_dict(save_obj['model_state_dict'])
    optimizer.load_state_dict(save_obj['optimizer_state_dict'])
    model.to(device)
    print(f"--- PyTorch Checkpoint loaded from {filename}, resuming from epoch {save_obj['epoch']} ---")
    return save_obj['epoch']
    
# ==============================================================================
# 3. MAIN SCRIPT LOGIC
# ==============================================================================
def pretokenize_data(args):
    output_dir = Path(args.output_dir); output_dir.mkdir(exist_ok=True, parents=True)
    console = Console() if Console else type("Dummy", (), {"print": lambda s, *a, **k: print(s)})()
    console.print(f"--- ⚡ [bold]Ultra-Fast Byte-Level Pre-processing[/bold] from [cyan]{args.data_path}[/cyan]... ---", style="bold yellow")
    with open(args.data_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
    all_bytes_np = np.frombuffer(content.encode('utf-8', errors='ignore'), dtype=np.uint8)
    
    data_path = output_dir / 'data.npy'; np.save(data_path, all_bytes_np)
    console.print(f"✅ Saved {len(all_bytes_np):,} raw bytes to [cyan]{data_path}[/cyan]")
    console.print("✅ Skipping pre-calculation of hashes. They will be computed on-the-fly during training.")
    
def train(args):
    # Model parameters
    HASH_WINDOW, D_MODEL, N_HEADS = 5, 512, 8
    NUM_LAYERS, D_FF, MAX_SEQ_LEN = 6, 2048, 4096
    K_SPARSE, WORKING_MEMORY_SIZE = 64, 128
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Phoenix Protocol v6.1 (WuBu Hybrid) ---"); print(f"--- Using device: {device.upper()} ---")
    
    data_path = Path(args.data_path) / 'data.npy'
    if not data_path.exists():
        print(f"FATAL: data.npy not found in {args.data_path}. Run `pretokenize-data` first."); sys.exit(1)
    
    corpus_bytes = np.load(data_path, mmap_mode='r')
    print(f"--- Corpus loaded via memory map: {len(corpus_bytes):,} bytes ---")

    model = WuBuMemoryWBA_Model(256, D_MODEL, NUM_LAYERS, N_HEADS, D_FF, K_SPARSE, WORKING_MEMORY_SIZE, MAX_SEQ_LEN).to(device)
    print(f'--- WuBu Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters. ---')
    
    num_devices = 1 
    effective_batch_size = args.batch_size * num_devices
    num_examples = len(corpus_bytes) - args.block_size - (HASH_WINDOW - 1)
    num_batches_per_epoch = num_examples // effective_batch_size if num_examples > effective_batch_size else 1
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    ckpt_file = Path(f"./{args.basename}.wubu.pkl")
    start_epoch = load_checkpoint_torch(model, optimizer, ckpt_file, device)
    
    epochs_to_run = args.epochs
    if start_epoch >= epochs_to_run and not args.fresh_start:
        print(f"Training previously completed for {start_epoch} epochs.")
        while True:
            try:
                extra = input(f"Train for more epochs? (Enter number, or 'q' to quit): ")
                if extra.lower() in ['q', 'quit']: return
                epochs_to_run = start_epoch + int(extra); break
            except ValueError: print("Invalid input.")

    if args.fresh_start: start_epoch = 0; print("--- Starting fresh training run (--fresh-start) ---")
    
    if num_batches_per_epoch > 0 and start_epoch < epochs_to_run:
        train_gen = data_generator(corpus_bytes, args.seed, args.batch_size, args.block_size, HASH_WINDOW, num_devices)
        start_time, epoch_for_interrupt = time.time(), start_epoch
        try:
            for epoch in range(start_epoch, epochs_to_run):
                epoch_for_interrupt = epoch
                with tqdm(range(num_batches_per_epoch), desc=f"Epoch {epoch+1}/{epochs_to_run}", leave=True) as pbar:
                    for i in pbar:
                        batch = next(train_gen)
                        loss = train_step_torch(model, optimizer, criterion, batch, device)
                        if i % 10 == 0: pbar.set_postfix(loss=f"{loss:.4f}")
                save_checkpoint_torch(model, optimizer, epoch + 1, ckpt_file)
            print(f"\nTraining finished in {time.time() - start_time:.2f}s")
        except (KeyboardInterrupt, StopIteration):
            save_checkpoint_torch(model, optimizer, epoch_for_interrupt, ckpt_file)
            print("\n--- Training interrupted or finished. State saved. ---")

def chat(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console = Console() if Console else type("Dummy", (), {"print": lambda s, *a, **k: print(s)})()
    console.print(f"--- 💬 Loading WuBu model [cyan]{args.model_path}[/cyan] ---", style="bold yellow")
    
    # These params must match the saved model
    D_MODEL, N_HEADS = 512, 8
    NUM_LAYERS, D_FF, MAX_SEQ_LEN = 6, 2048, 4096
    K_SPARSE, WORKING_MEMORY_SIZE = 64, 128

    model = WuBuMemoryWBA_Model(256, D_MODEL, NUM_LAYERS, N_HEADS, D_FF, K_SPARSE, WORKING_MEMORY_SIZE, MAX_SEQ_LEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    start_epoch = load_checkpoint_torch(model, optimizer, args.model_path, device)
    console.print(f"--- Loaded model trained for {start_epoch} epochs. ---")
    
    hasher = RollingHasher(window_size=5)
    md_print = Markdown if Markdown else lambda x: print(x)
    console.print(md_print("--- \n**Model loaded.** Type prompts below. Use `quit` or `exit` to end. \n---"))

    while True:
        prompt = input(">> ")
        if prompt.lower() in ["quit", "exit"]: break
        prompt_bytes = np.frombuffer(f"User: {prompt}\nAssistant: ".encode(), dtype=np.uint8)
        console.print("Assistant: ", end="", style="bold cyan")
        generated_bytes = list(prompt_bytes)
        for _ in range(1024):
            current_bytes = np.array(generated_bytes)
            indices_context = current_bytes[-args.block_size:]
            
            # Use the hasher to create hashes for the current context on the fly
            first_valid_idx = hasher.window_size - 1
            hash_src_context = current_bytes[-(args.block_size + first_valid_idx):]
            
            pad_len_idx = args.block_size - len(indices_context)
            if pad_len_idx > 0: indices_context = np.pad(indices_context, (pad_len_idx, 0), constant_values=32)
            
            pad_len_hash_src = args.block_size + first_valid_idx - len(hash_src_context)
            if pad_len_hash_src > 0: hash_src_context = np.pad(hash_src_context, (pad_len_hash_src, 0), constant_values=32)

            shape = (1, args.block_size, hasher.window_size)
            strides = (hash_src_context.strides[0], hash_src_context.strides[0], hash_src_context.strides[0])
            hash_windows = as_strided(hash_src_context, shape=shape, strides=strides)
            hashes_context = hasher.hash_batch(hash_windows).squeeze(0)

            indices_tensor = torch.from_numpy(indices_context).unsqueeze(0).to(device, dtype=torch.long)
            hashes_tensor = torch.from_numpy(hashes_context).unsqueeze(0).to(device)

            next_byte = predict_step_torch(model, indices_tensor, hashes_tensor)
            
            try:
                char = bytes([next_byte]).decode('utf-8')
                print(char, end="", flush=True)
            except UnicodeDecodeError: pass
            generated_bytes.append(next_byte)
            if next_byte == 10: 
                assistant_response = bytes(generated_bytes[len(prompt_bytes):]).decode('utf-8', 'ignore')
                if "User:" in assistant_response or "human:" in assistant_response: break
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phoenix Protocol v6.1: Fused GPT-at-Home & wubumind"); subparsers = parser.add_subparsers(dest="command", required=True)
    
    p_pretokenize = subparsers.add_parser("pretokenize-data", help="Pre-process a text file into raw bytes.")
    p_pretokenize.add_argument('--data-path', type=str, required=True, help="Path to the raw .txt file.")
    p_pretokenize.add_argument('--output-dir', type=str, required=True)
    
    p_train = subparsers.add_parser("pretrain", help="Train the WuBuMind model.")
    p_train.add_argument('--basename', type=str, required=True, help="Base name for model checkpoint files.")
    p_train.add_argument('--data-path', type=str, required=True, help="Path to the directory from 'pretokenize-data'.")
    p_train.add_argument('--epochs', type=int, default=20); p_train.add_argument('--block-size', type=int, default=512)
    p_train.add_argument('--batch-size', type=int, default=16, help="Per-device batch size.")
    p_train.add_argument('--learning-rate', type=float, default=5e-4); p_train.add_argument('--seed', type=int, default=42)
    p_train.add_argument('--fresh-start', action='store_true', help="Force a fresh start, ignoring existing checkpoints.")
    
    p_chat = subparsers.add_parser("chat", help="Chat with a trained WuBuMind model.")
    p_chat.add_argument('--model-path', type=str, required=True, help="Path to the .wubu.pkl checkpoint file.")
    p_chat.add_argument('--block-size', type=int, default=512)
    
    args = parser.parse_args()

    if args.command == "pretokenize-data": pretokenize_data(args)
    elif args.command == "pretrain": train(args)
    elif args.command == "chat": chat(args)