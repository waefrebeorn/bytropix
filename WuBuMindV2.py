# Gemini: The final challenge. The Mecca Script, WubuMind, is now fueled by a proper
# dataset: the complete works of Shakespeare.
#
# We have proven the architecture works. Now we give it a soul.
# This is the Grand Finale.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import math
import time
import os
import requests
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
    def expmap0(v,c): sqrt_c=torch.sqrt(c).unsqueeze(-1); v_norm=torch.norm(v,p=2,dim=-1,keepdim=True).clamp_min(1e-8); return ((1./(sqrt_c*v_norm))*torch.tanh(sqrt_c*v_norm))*v
    @staticmethod
    def dist(x,y,c): sqrt_c=torch.sqrt(c).squeeze(); arg=(1+2*c*torch.sum((x-y)**2,dim=-1)/((1-c*torch.sum(x**2,dim=-1))*(1-c*torch.sum(y**2,dim=-1))+1e-8)).clamp(min=1.0); return (1.0/sqrt_c)*torch.acosh(arg)

class kNNHyperbolicAttentionLayer(nn.Module):
    def __init__(self, dim, n_heads, k):
        super().__init__(); self.dim, self.n_heads, self.k, self.h_dim = dim, n_heads, k, dim // n_heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(dim, dim) for _ in range(4))
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim); self.log_tau = nn.Parameter(torch.tensor(0.0))
    def forward(self, x, positions, c):
        B, N, _ = x.shape; x_res = x; x_norm1 = self.norm1(x)
        q,k,v = (p(x_norm1).view(B,N,self.n_heads,self.h_dim).transpose(1,2) for p in (self.q_proj, self.k_proj, self.v_proj))
        with torch.no_grad():
            dist_matrix = PoincareBall.dist(positions.unsqueeze(1), positions.unsqueeze(2), c)
            attn_dists, top_k_indices = torch.topk(dist_matrix, self.k, dim=-1, largest=False)
        indices = top_k_indices.unsqueeze(1).unsqueeze(4).expand(-1, self.n_heads, -1, -1, self.h_dim)
        k_gathered, v_gathered = torch.gather(k.unsqueeze(3).expand(-1,-1,-1,self.k,-1),2,indices), torch.gather(v.unsqueeze(3).expand(-1,-1,-1,self.k,-1),2,indices)
        feature_scores = torch.matmul(q.unsqueeze(3), k_gathered.transpose(-1, -2)).squeeze(3) / math.sqrt(self.h_dim)
        geometric_scores = -(attn_dists.unsqueeze(1)) / (torch.exp(self.log_tau) + 1e-8)
        attn_probs = F.softmax(feature_scores + geometric_scores, dim=-1)
        attn_output = torch.matmul(attn_probs.unsqueeze(3), v_gathered).squeeze(3).transpose(1, 2).reshape(B, N, -1)
        x = x_res + self.out_proj(attn_output); x = x + self.ffn(self.norm2(x)); return x

# --- Part 3: The Mecca Script - WubuMind ---
class WubuMindDataset(Dataset):
    def __init__(self, hashes, indices, targets):
        self.hashes,self.indices,self.targets = torch.tensor(hashes,dtype=torch.int64),torch.tensor(indices,dtype=torch.int64),torch.tensor(targets,dtype=torch.int64)
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx): return self.hashes[idx], self.indices[idx], self.targets[idx]

class WubuMind(nn.Module):
    def __init__(self, ctx_len, vocab_size, d_model, n_heads, n_layers, k, mod, c=1.0, drop=0.2):
        super().__init__(); self.context_length, self.modulus, self.d_model = ctx_len, float(mod), d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model); self.hash_projector = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.bridge_proj = nn.Linear(2 * d_model, d_model)
        self.log_c = nn.Parameter(torch.tensor(math.log(c))); self.hyperbolic_positions_tangent = nn.Parameter(torch.randn(ctx_len, d_model) * 0.02)
        self.hyperbolic_layers = nn.ModuleList([kNNHyperbolicAttentionLayer(d_model, n_heads, k) for _ in range(n_layers)])
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, hashes, indices):
        B = hashes.shape[0]; char_embed = self.token_embedding(indices)
        hash_embed = (hashes.unsqueeze(-1) / self.modulus).float() @ self.hash_projector
        x = self.bridge_proj(torch.cat([char_embed, hash_embed], dim=-1))
        c = torch.exp(self.log_c); current_positions = PoincareBall.expmap0(self.hyperbolic_positions_tangent, c).expand(B, -1, -1)
        for layer in self.hyperbolic_layers: x = layer(x, current_positions, c)
        return self.output_proj(x[:, -1, :])

    @torch.no_grad()
    def generate(self, prompt, steps, temp=0.6, top_p=0.9):
        self.eval(); device = next(self.parameters()).device
        text, values, indices = prompt, ascii_converter.convert(prompt), ascii_converter.get_indices(prompt)
        min_len = self.context_length + HASH_WINDOW - 1
        if len(indices) < min_len:
            padding = [ascii_converter.char_to_idx.get(' ', 0)] * (min_len - len(indices))
            indices, values = padding + indices, [ascii_converter.char_to_val.get(' ', 0)] * (min_len - len(values)) + values
        for _ in range(steps):
            context_hashes_np = hasher.hash_sequence(values[-min_len:])
            context_indices_np = indices[-self.context_length:]
            hashes_t, indices_t = torch.tensor([context_hashes_np],dtype=torch.int64).to(device), torch.tensor([context_indices_np],dtype=torch.int64).to(device)
            logits = self(hashes_t, indices_t); logits = logits.squeeze() / temp; probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True); cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p; sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone(); sorted_indices_to_remove[0] = 0
            probs[sorted_indices[sorted_indices_to_remove]] = 0.0; probs /= torch.sum(probs)
            next_idx = torch.multinomial(probs, num_samples=1).item(); next_char = ascii_converter.idx_to_char.get(next_idx, ' ')
            text += next_char; values.append(ascii_converter.char_to_val.get(next_char, 0)); indices.append(ascii_converter.char_to_idx.get(next_char, 0))
            if len(text) > 1500: break
        self.train(); return text

# --- Main Execution Block ---
if __name__ == "__main__":
    CONTEXT_LENGTH, HASH_WINDOW = 128, 5
    D_MODEL, N_HEADS, N_LAYERS, K_NEIGHBORS = 384, 6, 6, 24
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 1e-3, 32, 64
    MODULUS, MODEL_FILE = 10**9 + 7, "wubumind_v2_shakespeare.pth"
    DATA_URL, DATA_FILE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "tinyshakespeare.txt"
    FORCE_RETRAIN = True

    device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"--- Using device: {device} ---")
    torch.manual_seed(42); np.random.seed(42)
    if device == "cuda": torch.cuda.manual_seed(42)

    if not os.path.exists(DATA_FILE):
        print(f"Downloading dataset..."); r=requests.get(DATA_URL); r.raise_for_status();
        with open(DATA_FILE,'w',encoding='utf-8') as f: f.write(r.text)
        print("Download complete.")
    
    with open(DATA_FILE,'r',encoding='utf-8') as f: corpus_text=f.read()
    
    ascii_converter = SimplifiedASCIIConverter(corpus_text)
    hasher = RollingHasher(window_size=HASH_WINDOW, modulus=MODULUS)
    model = WubuMind(CONTEXT_LENGTH,ascii_converter.vocab_size,D_MODEL,N_HEADS,N_LAYERS,K_NEIGHBORS,MODULUS).to(device)
    print(f"--- WubuMind Initialized ---"); print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")

    if os.path.exists(MODEL_FILE) and not FORCE_RETRAIN:
        model.load_state_dict(torch.load(MODEL_FILE)); print(f"--- WubuMind weights loaded from {MODEL_FILE} ---")
    else:
        print(f"--- Training WubuMind on '{DATA_FILE}' ({len(corpus_text):,} chars)... ---")
        values=ascii_converter.convert(corpus_text); hashes=hasher.hash_sequence(values); indices=ascii_converter.get_indices(corpus_text)
        h, ind, t = [], [], []
        for i in range(len(indices) - CONTEXT_LENGTH - HASH_WINDOW):
            h.append(hashes[i+1:i+CONTEXT_LENGTH+1]); ind.append(indices[i+HASH_WINDOW:i+CONTEXT_LENGTH+HASH_WINDOW]); t.append([indices[i+CONTEXT_LENGTH+HASH_WINDOW]])

        dataset = WubuMindDataset(np.array(h), np.array(ind), np.array(t))
        dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,pin_memory=True,prefetch_factor=2)
        optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time(); model.train()
        for epoch in range(EPOCHS):
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
            for batch_hashes, batch_indices, batch_targets in progress_bar:
                batch_hashes, batch_indices, batch_targets = batch_hashes.to(device,non_blocking=True),batch_indices.to(device,non_blocking=True),batch_targets.squeeze().to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_hashes, batch_indices), batch_targets); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        print(f"\nTraining finished in {time.time() - start_time:.2f}s\n")
        torch.save(model.state_dict(), MODEL_FILE); print(f"--- WubuMind weights saved to {MODEL_FILE} ---")

    print(f"\n--- Generating from the Grand Finale: WubuMind ---")
    prompt = "Shall I compare thee to a summer's day?"
    print(f"Prompt: '{prompt}'\nResult:")
    print(model.generate(prompt, steps=1200))
    print("\n" + "="*50 + "\n")
    prompt = "To be, or not to be, that is the question:"
    print(f"Prompt: '{prompt}'\nResult:")
    print(model.generate(prompt, steps=1200))