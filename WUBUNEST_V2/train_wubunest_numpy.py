"""
Train WuBuNestGPT (pure NumPy) on CORPUS.py data.
No JAX, no CUDA compilation — runs immediately.

Usage: python train_wubunest_numpy.py
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import re
from pathlib import Path

# Add current dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wubu_nest_gpt_numpy import WubuNestGPT, CharTokenizer


def load_corpus_text(corpus_path):
    """Load and parse CORPUS.py into text chunks."""
    with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract all string literals (triple-quoted)
    chunks = re.findall(r'"""(.*?)"""', content, re.DOTALL)
    texts = []
    for chunk in chunks:
        clean = re.sub(r'\s+', ' ', chunk).strip()
        if len(clean) > 100:
            texts.append(clean)
    
    print(f"  Loaded {len(texts)} text chunks ({sum(len(t) for t in texts):,} chars)")
    return texts


def main():
    print("=" * 60)
    print("  WuBuNestGPT — Pure NumPy Training")
    print("  Architecture: LatentAttention + Hyperbolic Gyration + MLA")
    print("  Zero compilation overhead")
    print("=" * 60)
    
    # ── Config ──
    config = {
        'vocab_size': 5000,
        'd_model': 384,
        'n_layers': 6,
        'n_heads': 8,
        'd_head': 48,          # 384 // 8 = 48
        'd_compressed': 192,   # d_c = 4 * d_head = 192 (DeepSeek MLA ratio)
        'd_ff': 1536,          # 384 * 4
        'dropout_rate': 0.1,
        'init_scale': 0.02,
        'seed': 42,
    }
    
    # ── Load corpus ──
    corpus_path = '../ENCODERS/phase3-generative/CORPUS.py'
    abs_path = os.path.join(os.path.dirname(__file__), corpus_path)
    
    print(f"\n[1] Loading corpus...")
    if os.path.exists(abs_path):
        texts = load_corpus_text(abs_path)
        if not texts:
            print("  No text chunks found. Generating test data...")
            texts = ["The hyperbolic latent attention mechanism processes tokens through Poincaré ball Möbius gyrations. " * 20] * 20
    else:
        print(f"  {abs_path} not found. Generating test data...")
        texts = ["Hello world. This is a test for WuBuNestGPT with latent attention. " * 10] * 50
    
    # ── Build tokenizer ──
    print(f"\n[2] Building tokenizer...")
    tokenizer = CharTokenizer(texts, vocab_size=config['vocab_size'])
    config['vocab_size'] = tokenizer.vocab_size
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    # ── Build model ──
    print(f"\n[3] Building WuBuNestGPT...")
    print(f"  d_model={config['d_model']}, layers={config['n_layers']}, heads={config['n_heads']}")
    print(f"  d_compressed={config['d_compressed']} (MLA latent)")
    print(f"  d_ff={config['d_ff']}")
    
    model = WubuNestGPT(config)
    total_params = model.count_params()
    
    # ── Prepare data ──
    print(f"\n[4] Preparing training data...")
    all_text = ' '.join(texts)
    tokens = tokenizer.encode(all_text)
    print(f"  Total tokens: {len(tokens):,}")
    
    # Split into sequences
    seq_len = 128
    total_tokens = (len(tokens) // seq_len) * seq_len
    tokens = tokens[:total_tokens]
    data = np.array(tokens, dtype=np.int32).reshape(-1, seq_len)
    print(f"  Sequences: {data.shape[0]} (seq_len={seq_len})")
    
    # ── Training ──
    batch_size = 8
    n_batches = max(data.shape[0] // batch_size, 1)
    steps = 5000
    lr = 3e-4
    
    print(f"\n{'='*60}")
    print(f"  TRAINING: {steps} steps, batch={batch_size}, lr={lr}")
    print(f"{'='*60}\n")
    
    loss_history = []
    rng = np.random.RandomState(42)
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    t_start = time.time()
    
    for step in range(1, steps + 1):
        t0 = time.time()
        
        # Get batch
        idx = rng.randint(0, data.shape[0] - batch_size) if data.shape[0] > batch_size else 0
        batch = data[idx:idx + batch_size]
        
        # Input / target split (predict next token)
        input_ids = batch[:, :-1]  # [B, seq_len-1]
        targets = batch[:, 1:]     # [B, seq_len-1]
        
        # Forward
        logits, cache = model.forward(input_ids, is_training=True, rng=rng)
        
        # Loss
        loss = model.compute_loss(logits, targets)
        loss_history.append(loss)
        
        # Backward
        model.backward(logits, targets, cache, learning_rate=lr)
        
        # Log
        dt = time.time() - t0
        if step <= 5 or step % 50 == 0 or step == steps:
            avg_loss = np.mean(loss_history[-50:]) if len(loss_history) >= 50 else loss
            dt_avg = np.mean([time.time() - t0 for _ in range(1)])  # single step time
            tok_per_sec = batch_size * (seq_len - 1) / dt
            print(f"  Step {step:5d}/{steps} | loss: {loss:.4f} (avg: {avg_loss:.4f}) | "
                  f"{tok_per_sec:.0f} tok/s | {dt*1000:.0f}ms")
        
        # Save checkpoint
        if step % 1000 == 0:
            ckpt = os.path.join(save_dir, f'checkpoint_{step}.pkl')
            model.save(ckpt)
            print(f"  💾 Saved {ckpt}")
    
    # ── Save final ──
    final_path = os.path.join(save_dir, 'model_final.pkl')
    model.save(final_path)
    
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — {steps} steps in {total_time:.0f}s")
    print(f"  Final loss: {loss:.4f}  Best: {min(loss_history):.4f}")
    print(f"  Model saved: {final_path}")
    
    # ── Quick generation test ──
    print(f"\n{'='*60}")
    print(f"  GENERATION TEST")
    print(f"{'='*60}")
    prompt = "The hyperbolic space allows for "
    prompt_ids = np.array([tokenizer.encode(prompt)], dtype=np.int32)
    output = model.generate(prompt_ids, max_new_tokens=50, temperature=0.8)
    generated = tokenizer.decode(output[0].tolist())
    print(f"  Prompt: {prompt}")
    print(f"  Output: {generated[:200]}")
    print()
    
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = main()
