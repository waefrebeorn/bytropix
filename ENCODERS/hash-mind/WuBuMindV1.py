# Gemini: The Challenge is accepted.
#
# I have fused the dual-source embedding of HashMind with the hyperbolic geometry of WubuDiffusion.
# The standard Transformer has been replaced. The model now thinks in a stack of "WuBu Spheres."
# It is a new architecture, born from your vision.
#
# This is WubuMind.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import math
import time
import os
from tqdm import tqdm
import torch.nn.functional as F

# --- Part 1: HashMind's Input Engine (Tokenizer & Hasher) ---
class SimplifiedASCIIConverter:
    def __init__(self, corpus=""):
        self.char_to_val, self.val_to_char, self.char_to_idx, self.idx_to_char = {}, {}, {}, {}
        chars = sorted(list(set(corpus)))
        self.vocab_size = len(chars)
        for i, char in enumerate(chars):
            self.char_to_val[char], self.val_to_char[ord(char)] = ord(char), char
            self.char_to_idx[char], self.idx_to_char[i] = i, char
    def convert(self, text): return [self.char_to_val.get(c, 0) for c in text]
    def get_indices(self, text): return [self.char_to_idx.get(c, 0) for c in text]

class RollingHasher:
    def __init__(self, window_size, base=31, modulus=10**9 + 7):
        self.window_size, self.base, self.modulus, self.precomputed_base = window_size, base, modulus, pow(base, window_size - 1, modulus)
    def hash_sequence(self, values):
        if len(values) < self.window_size: return []
        hashes, current_hash = [], 0
        for i in range(self.window_size): current_hash = (current_hash * self.base + values[i]) % self.modulus
        hashes.append(current_hash)
        for i in range(1, len(values) - self.window_size + 1):
            old_val, new_val = values[i-1], values[i+self.window_size-1]
            current_hash = ((current_hash - old_val * self.precomputed_base) * self.base + new_val) % self.modulus
            if current_hash < 0: current_hash += self.modulus
            hashes.append(current_hash)
        return hashes

# --- Part 2: WuBu's Geometric Core (Hyperbolic Components) ---
class PoincareBall:
    @staticmethod
    def expmap0(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sqrt_c = torch.sqrt(c).unsqueeze(-1)
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        lam = (1. / (sqrt_c * v_norm)) * torch.tanh(sqrt_c * v_norm)
        return lam * v
    @staticmethod
    def dist(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sqrt_c = torch.sqrt(c).squeeze()
        diff_norm_sq = torch.sum((x - y) ** 2, dim=-1)
        x_norm_sq = torch.sum(x ** 2, dim=-1)
        y_norm_sq = torch.sum(y ** 2, dim=-1)
        num = 2 * c * diff_norm_sq
        den = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
        arg = (1 + num / (den + 1e-8)).clamp(min=1.0)
        return (1.0 / sqrt_c) * torch.acosh(arg)

class kNNHyperbolicAttentionLayer(nn.Module):
    def __init__(self, dim, n_heads, k):
        super().__init__()
        self.dim, self.n_heads, self.k, self.h_dim = dim, n_heads, k, dim // n_heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(dim, dim) for _ in range(4))
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.log_tau = nn.Parameter(torch.tensor(0.0))
    def forward(self, x, positions, c):
        B, N, _ = x.shape
        x_res = x
        x_norm1 = self.norm1(x)
        q = self.q_proj(x_norm1).view(B, N, self.n_heads, self.h_dim).transpose(1, 2)
        k = self.k_proj(x_norm1).view(B, N, self.n_heads, self.h_dim).transpose(1, 2)
        v = self.v_proj(x_norm1).view(B, N, self.n_heads, self.h_dim).transpose(1, 2)
        with torch.no_grad():
            dist_matrix = PoincareBall.dist(positions.unsqueeze(1), positions.unsqueeze(2), c)
            attn_dists, top_k_indices = torch.topk(dist_matrix, self.k, dim=-1, largest=False)
        k_for_gather, v_for_gather = k.unsqueeze(3).expand(-1, -1, -1, self.k, -1), v.unsqueeze(3).expand(-1, -1, -1, self.k, -1)
        indices = top_k_indices.unsqueeze(1).unsqueeze(4).expand(-1, self.n_heads, -1, -1, self.h_dim)
        k_gathered, v_gathered = torch.gather(k_for_gather, 2, indices), torch.gather(v_for_gather, 2, indices)
        feature_scores = torch.matmul(q.unsqueeze(3), k_gathered.transpose(-1, -2)).squeeze(3) / math.sqrt(self.h_dim)
        tau = torch.exp(self.log_tau) + 1e-8
        geometric_scores = -(attn_dists.unsqueeze(1)) / tau
        attn_probs = F.softmax(feature_scores + geometric_scores, dim=-1)
        attn_output = torch.matmul(attn_probs.unsqueeze(3), v_gathered).squeeze(3).transpose(1, 2).reshape(B, N, -1)
        x = x_res + self.out_proj(attn_output)
        x = x + self.ffn(self.norm2(x))
        return x

# --- Part 3: The Mecca Script - WubuMind ---
class WubuMindDataset(Dataset):
    def __init__(self, hashes, indices, targets):
        self.hashes, self.indices, self.targets = torch.tensor(hashes, dtype=torch.int64), torch.tensor(indices, dtype=torch.int64), torch.tensor(targets, dtype=torch.int64)
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx): return self.hashes[idx], self.indices[idx], self.targets[idx]

class WubuMind(nn.Module):
    def __init__(self, context_length, vocab_size, d_model, n_heads, n_layers, k_neighbors, modulus, poincare_c=1.0, dropout=0.1):
        super().__init__()
        self.context_length, self.modulus, self.d_model = context_length, float(modulus), d_model
        # HashMind's Input
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.hash_projector = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.bridge_proj = nn.Linear(2 * d_model, d_model)
        # WuBu's Geometric Core
        self.log_c = nn.Parameter(torch.tensor(math.log(poincare_c))) # Curvature
        self.hyperbolic_positions_tangent = nn.Parameter(torch.randn(context_length, d_model) * 0.02) # Learnable positions
        self.hyperbolic_layers = nn.ModuleList([kNNHyperbolicAttentionLayer(d_model, n_heads, k_neighbors) for _ in range(n_layers)])
        # Output
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, hashes, indices):
        B = hashes.shape[0]
        # HashMind input processing
        char_embed = self.token_embedding(indices)
        hash_embed = (hashes.unsqueeze(-1) / self.modulus).float() @ self.hash_projector
        x = self.bridge_proj(torch.cat([char_embed, hash_embed], dim=-1))
        # WuBu Sphere processing
        c = torch.exp(self.log_c)
        current_positions = PoincareBall.expmap0(self.hyperbolic_positions_tangent, c).expand(B, -1, -1)
        for layer in self.hyperbolic_layers:
            x = layer(x, current_positions, c)
        return self.output_proj(x[:, -1, :])

    @torch.no_grad()
    def generate(self, prompt, steps, temperature=0.6, top_p=0.9):
        self.eval(); device = next(self.parameters()).device
        text, values, indices = prompt, ascii_converter.convert(prompt), ascii_converter.get_indices(prompt)
        min_len = self.context_length + HASH_WINDOW - 1
        if len(indices) < min_len:
            padding = [ascii_converter.char_to_idx.get(' ', 0)] * (min_len - len(indices))
            indices, values = padding + indices, [ascii_converter.char_to_val.get(' ', 0)] * (min_len - len(values)) + values
        for _ in range(steps):
            context_hashes_np = hasher.hash_sequence(values[-min_len:])
            context_indices_np = indices[-self.context_length:]
            hashes_tensor = torch.tensor([context_hashes_np], dtype=torch.int64).to(device)
            indices_tensor = torch.tensor([context_indices_np], dtype=torch.int64).to(device)
            logits = self(hashes_tensor, indices_tensor)
            logits = logits.squeeze() / temperature; probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True); cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p; sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone(); sorted_indices_to_remove[0] = 0
            probs[sorted_indices[sorted_indices_to_remove]] = 0.0; probs /= torch.sum(probs)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = ascii_converter.idx_to_char.get(next_idx, ' ')
            text += next_char; values.append(ascii_converter.char_to_val.get(next_char, 0)); indices.append(ascii_converter.char_to_idx.get(next_char, 0))
            if len(text) > 1000: break
        self.train(); return text

# --- Main Execution Block ---
if __name__ == "__main__":
    CONTEXT_LENGTH, HASH_WINDOW = 64, 5
    D_MODEL, N_HEADS, N_LAYERS, K_NEIGHBORS = 256, 4, 4, 16 # WuBu Sphere Hyperparams
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 3e-4, 64, 50
    MODULUS, MODEL_FILE = 10**9 + 7, "wubumind_v1_selfaware.pth"
    FORCE_RETRAIN = False

    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"--- Using device: {device} ---")
    torch.manual_seed(42); np.random.seed(42)
    if device == "cuda": torch.cuda.manual_seed(42)

    try:
        with open(__file__, 'r', encoding='utf-8') as f: full_code = f.read()
        corpus_text = full_code.split('if __name__ == "__main__":')[0]
    except Exception as e:
        print(f"Could not read source file ({e}), using default text.")
        corpus_text = "This is a fallback text for training WubuMind."

    ascii_converter = SimplifiedASCIIConverter(corpus_text)
    hasher = RollingHasher(window_size=HASH_WINDOW, modulus=MODULUS)
    model = WubuMind(CONTEXT_LENGTH, ascii_converter.vocab_size, D_MODEL, N_HEADS, N_LAYERS, K_NEIGHBORS, MODULUS).to(device)
    print(f"--- WubuMind Initialized ---")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")

    if os.path.exists(MODEL_FILE) and not FORCE_RETRAIN:
        model.load_state_dict(torch.load(MODEL_FILE)); print(f"--- WubuMind weights loaded from {MODEL_FILE} ---")
    else:
        print(f"--- Training WubuMind on its own source code ({len(corpus_text):,} chars)... ---")
        values = ascii_converter.convert(corpus_text); hashes = hasher.hash_sequence(values); indices = ascii_converter.get_indices(corpus_text)
        all_input_hashes, all_input_indices, all_targets = [], [], []
        for i in range(len(indices) - CONTEXT_LENGTH - HASH_WINDOW):
            all_input_hashes.append(hashes[i+1 : i+CONTEXT_LENGTH+1])
            all_input_indices.append(indices[i+HASH_WINDOW : i+CONTEXT_LENGTH+HASH_WINDOW])
            all_targets.append([indices[i + CONTEXT_LENGTH + HASH_WINDOW]])

        dataset = WubuMindDataset(np.array(all_input_hashes), np.array(all_input_indices), np.array(all_targets))
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time(); model.train()
        for epoch in range(EPOCHS):
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
            for batch_hashes, batch_indices, batch_targets in progress_bar:
                batch_hashes, batch_indices, batch_targets = batch_hashes.to(device), batch_indices.to(device), batch_targets.squeeze().to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_hashes, batch_indices), batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            avg_loss = sum(p['loss'] for p in progress_bar.postfixes) / len(progress_bar.postfixes)
            print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")

        print(f"\nTraining finished in {time.time() - start_time:.2f}s\n")
        torch.save(model.state_dict(), MODEL_FILE); print(f"--- WubuMind weights saved to {MODEL_FILE} ---")

    print(f"\n--- Generating from the Mecca Script: WubuMind ---")
    prompt = "class WubuMind(nn.Module):"
    print(f"Prompt: '{prompt}'\nResult:")
    print(model.generate(prompt, steps=500))
    print("\n" + "="*50 + "\n")
    prompt = "def forward(self, hashes, indices):"
    print(f"Prompt: '{prompt}'\nResult:")
    print(model.generate(prompt, steps=500))