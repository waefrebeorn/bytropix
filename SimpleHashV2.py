# Gemini: The model has cleverly overfit to the print statements at the end of the script.
# This is a data quality problem. The fix is to train it only on the 'pure' code.
# I've modified the script to automatically use only the code *before* the main execution
# block for training. This will force it to learn the real structure of Python.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import math
import time
import os

# --- Part 1: Tokenizer & Hasher (Unchanged) ---
class SimplifiedASCIIConverter:
    def __init__(self):
        self.char_to_val, self.val_to_char, self.char_to_idx, self.idx_to_char = {}, {}, {}, {}
        self.vocab_size = 0
        chars = ([str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)] +
                 [chr(ord('a') + i) for i in range(26)] + [' ', '.', ',', '!', '?', '\n', '#', '=', '-', '(', ')', ':', "'", '_', '[', ']', '{', '}', '<', '>', '/', '\\', '*'])
        for char in sorted(list(set(chars))): self._add_char(char, ord(char))
    def _add_char(self, char, val):
        if char not in self.char_to_val: self.char_to_val[char], self.val_to_char[val], self.char_to_idx[char], self.idx_to_char[self.vocab_size] = val, char, self.vocab_size, char; self.vocab_size += 1
    def convert(self, text): return [self.char_to_val.get(c, self.char_to_val[' ']) for c in text]
    def get_indices(self, text): return [self.char_to_idx.get(c, self.char_to_idx[' ']) for c in text]

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

# --- Part 2: The Neural Network (V14 - Curated Data) ---

class HashMindDataset(Dataset):
    def __init__(self, hashes, indices, targets):
        self.hashes, self.indices, self.targets = torch.tensor(hashes, dtype=torch.int64), torch.tensor(indices, dtype=torch.int64), torch.tensor(targets, dtype=torch.int64)
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx): return self.hashes[idx], self.indices[idx], self.targets[idx]

class HashMind(nn.Module):
    def __init__(self, context_length, vocab_size, d_model, n_heads, n_layers, modulus, dropout=0.1):
        super().__init__()
        self.context_length, self.modulus, self.d_model = context_length, float(modulus), d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.hash_projector = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.bridge_proj = nn.Linear(2 * d_model, d_model)
        pe = torch.zeros(context_length, d_model); position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, hashes, indices):
        char_embed = self.token_embedding(indices)
        hash_embed = (hashes.unsqueeze(-1) / self.modulus).float() @ self.hash_projector
        x = self.bridge_proj(torch.cat([char_embed, hash_embed], dim=-1)) + self.pe
        x = self.transformer_encoder(x)
        return self.output_proj(x[:, -1, :])

    @torch.no_grad()
    def generate(self, prompt, steps, temperature=0.6, top_p=0.9):
        self.eval(); device = next(self.parameters()).device
        text, values, indices = prompt, ascii_converter.convert(prompt), ascii_converter.get_indices(prompt)
        min_len = self.context_length + HASH_WINDOW - 1
        if len(indices) < min_len:
            padding = [ascii_converter.char_to_idx[' ']] * (min_len - len(indices))
            indices = padding + indices
            values = [ascii_converter.char_to_val[' ']] * (min_len - len(values)) + values
        for _ in range(steps):
            context_hashes_np = hasher.hash_sequence(values[-min_len:])
            context_indices_np = indices[-self.context_length:]
            hashes_tensor = torch.tensor([context_hashes_np], dtype=torch.int64).to(device)
            indices_tensor = torch.tensor([context_indices_np], dtype=torch.int64).to(device)
            logits = self(hashes_tensor, indices_tensor)
            logits = logits.squeeze() / temperature
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0.0
            probs /= torch.sum(probs)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = ascii_converter.idx_to_char.get(next_idx, ' ')
            text += next_char
            values.append(ascii_converter.char_to_val.get(next_char, ord(' ')))
            indices.append(ascii_converter.char_to_idx.get(next_char, 0))
            if len(text) > 500: break
        self.train(); return text

# --- Main Execution Block ---
if __name__ == "__main__":
    CONTEXT_LENGTH, HASH_WINDOW = 64, 5
    D_MODEL, N_HEADS, N_LAYERS = 256, 4, 4
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 3e-4, 64, 50
    MODULUS, MODEL_FILE = 10**9 + 7, "hashmind_v14_pytorch.pth"
    FORCE_RETRAIN = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device} ---")
    torch.manual_seed(42); np.random.seed(42)
    if device == "cuda": torch.cuda.manual_seed(42)
    ascii_converter = SimplifiedASCIIConverter()
    hasher = RollingHasher(window_size=HASH_WINDOW, modulus=MODULUS)
    model = HashMind(CONTEXT_LENGTH, ascii_converter.vocab_size, D_MODEL, N_HEADS, N_LAYERS, MODULUS).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")

    if os.path.exists(MODEL_FILE) and not FORCE_RETRAIN:
        model.load_state_dict(torch.load(MODEL_FILE))
        print(f"--- Model loaded from {MODEL_FILE} ---")
    else:
        print("--- Training new model with curated data... ---")
        try:
            with open(__file__, 'r', encoding='utf-8') as f: full_code = f.read()
            # --- THE FIX: DATA CURATION ---
            corpus_text = full_code.split('if __name__ == "__main__":')[0]
            print(f"--- Training on first {len(corpus_text)} chars of source code (up to main block). ---")
        except:
            corpus_text, _ = "The quick brown fox jumps over the lazy dog.", print("--- Could not read file, training on default text. ---")
        
        values = ascii_converter.convert(corpus_text); hashes = hasher.hash_sequence(values); indices = ascii_converter.get_indices(corpus_text)
        all_input_hashes, all_input_indices, all_targets = [], [], []
        for i in range(len(indices) - CONTEXT_LENGTH - HASH_WINDOW):
            target_pos = i + CONTEXT_LENGTH + HASH_WINDOW
            hash_start, hash_end = i + 1, i + CONTEXT_LENGTH + 1
            char_start, char_end = i + HASH_WINDOW, i + CONTEXT_LENGTH + HASH_WINDOW
            all_input_hashes.append(hashes[hash_start:hash_end])
            all_input_indices.append(indices[char_start:char_end])
            all_targets.append([indices[target_pos]])

        dataset = HashMindDataset(np.array(all_input_hashes), np.array(all_input_indices), np.array(all_targets))
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time(); model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for batch_hashes, batch_indices, batch_targets in dataloader:
                batch_hashes, batch_indices, batch_targets = batch_hashes.to(device, non_blocking=True), batch_indices.to(device, non_blocking=True), batch_targets.squeeze().to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_hashes, batch_indices), batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {epoch_loss / len(dataloader):.4f}")

        print(f"Training finished in {time.time() - start_time:.2f}s\n")
        torch.save(model.state_dict(), MODEL_FILE); print(f"--- Model saved to {MODEL_FILE} ---")

    print(f"\n--- Generating from final trained model ---")
    prompt = "class HashMind(nn.Module):"
    print(f"Prompt: '{prompt}'\nResult:")
    print(model.generate(prompt, steps=300))
    print("\n" + "="*50 + "\n")
    prompt = "def __init__(self, context_length, vocab_size, d_model,"
    print(f"Prompt: '{prompt}'\nResult:")
    print(model.generate(prompt, steps=300))