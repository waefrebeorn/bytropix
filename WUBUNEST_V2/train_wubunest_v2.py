"""
train_wubunest_v2.py: Training loop for WubuNestGPT v2

Loads CORPUS.py, creates tokenizer, trains the model with:
- Cross-entropy language modeling loss
- Auto-regressive next-token prediction
- KV cache turbo quantization enabled during inference
- Auxiliary-loss-free MoE load balancing
- Hyperbolic gyration position encoding

Usage: python train_wubunest_v2.py [--corpus ../ENCODERS/phase3-generative/CORPUS.py] [--steps 5000]
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Settable flags for CPU fallback
os.environ['JAX_PLATFORMS'] = os.environ.get('JAX_PLATFORMS', '')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

from wubu_nest_gpt_v2 import WubuNestGPT, create_train_state, load_corpus_data, SimpleTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Train WubuNestGPT v2')
    parser.add_argument('--corpus', type=str, 
                        default='../ENCODERS/phase3-generative/CORPUS.py',
                        help='Path to CORPUS.py training data')
    parser.add_argument('--steps', type=int, default=5000,
                        help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--seq-len', type=int, default=256,
                        help='Sequence length for training')
    parser.add_argument('--d-model', type=int, default=384,
                        help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=6,
                        help='Number of transformer blocks')
    parser.add_argument('--n-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n-routed', type=int, default=16,
                        help='Number of MoE experts')
    parser.add_argument('--top-k', type=int, default=2,
                        help='Top-k experts per token')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--warmup', type=int, default=100,
                        help='Warmup steps')
    parser.add_argument('--vocab-size', type=int, default=10000,
                        help='Vocabulary size')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--no-quant', action='store_true',
                        help='Disable turbo quantization')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training')
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.cpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    
    print("=" * 60)
    print("  WubuNestGPT v2 — Training")
    print("  Architecture: LatentAttention + SparseMoE + Hyperbolic")
    print("=" * 60)
    
    # ── Config ──
    config = {
        'vocab_size': args.vocab_size,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'd_head': args.d_model // args.n_heads,
        'd_compressed': (args.d_model // args.n_heads) * 4,  # d_c = 4*d_h
        'd_ff': args.d_model * 4,
        'n_shared': 2,
        'n_routed': args.n_routed,
        'top_k': args.top_k,
        'use_quant': not args.no_quant,
        'dropout_rate': 0.1,
        'dtype': 'bfloat16',
        'learning_rate': args.lr,
        'warmup_steps': args.warmup,
        'train_steps': args.steps,
        'weight_decay': 0.1,
        'aux_loss_coef': 0.01,
        'max_seq_len': args.seq_len,
    }
    
    # ── Create tokenizer ──
    print(f"\n[1/4] Creating tokenizer (vocab_size={args.vocab_size})...")
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    print(f"  Tokenizer ready. Vocab: {tokenizer.get_vocab_size()}")
    
    # ── Load corpus ──
    print(f"\n[2/4] Loading corpus from {args.corpus}...")
    corpus_path = os.path.join(os.path.dirname(__file__), args.corpus)
    if not os.path.exists(corpus_path):
        print(f"  WARNING: {corpus_path} not found. Using test data instead.")
        texts = ["Hello world. This is a test document for WuBuNestGPT."] * 50
    else:
        texts = load_corpus_data(corpus_path)
        if len(texts) == 0:
            print("  No text chunks found. Using fallback data.")
            texts = ["The hyperbolic latent attention mechanism processes tokens through Poincaré ball Möbius gyrations."] * 100
    
    print(f"  Loaded {len(texts)} text chunks")
    
    # ── Create model ──
    print(f"\n[3/4] Creating WubuNestGPT v2...")
    print(f"  d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")
    print(f"  MoE: {args.n_routed} experts, top-{args.top_k} per token")
    print(f"  Latent attention: d_c={config['d_compressed']}, d_h={config['d_head']}")
    print(f"  Turbo quant: {'ON' if config['use_quant'] else 'OFF'}")
    
    model = WubuNestGPT(config=config)
    
    # Initialize and count params
    rng = jax.random.PRNGKey(42)
    dummy = jnp.zeros((2, 16), dtype=jnp.int32)
    variables = model.init(rng, dummy, is_training=True)
    param_count = sum(x.size for x in jax.tree.leaves(variables))
    print(f"  Total parameters: {param_count:,}")
    
    # ── Prepare data ──
    print(f"\n[4/4] Preparing training data...")
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
    
    print(f"  Total tokens: {len(all_tokens):,}")
    
    # Cut into sequences
    seq_len = args.seq_len
    total_tokens = (len(all_tokens) // seq_len) * seq_len
    all_tokens = all_tokens[:total_tokens]
    tokens_arr = np.array(all_tokens, dtype=np.int32)
    tokens_arr = tokens_arr.reshape(-1, seq_len)
    n_samples = tokens_arr.shape[0]
    print(f"  Sequences: {n_samples} (seq_len={seq_len})")
    
    # ── Create optimizer and train state ──
    print("\n  Initializing optimizer (AdamW with cosine decay)...")
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=args.warmup,
        decay_steps=args.steps,
        end_value=args.lr * 0.01
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            weight_decay=config['weight_decay']
        )
    )
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )
    
    # ── JIT loss function ──
    dropout_rng = jax.random.PRNGKey(42)
    
    @jax.jit
    def compute_loss(params, input_ids, labels, dropout_rng):
        """Cross-entropy loss."""
        logits, _, _ = model.apply(
            {'params': params},
            input_ids,
            is_training=True,
            rngs={'dropout': dropout_rng}
        )
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V).astype(jnp.float32)
        labels_flat = labels.reshape(-1)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, labels_flat)
        return loss.mean()
    
    @jax.jit
    def train_step(state, input_ids, labels, dropout_rng):
        """Single training step with gradient computation."""
        grad_fn = jax.value_and_grad(compute_loss)
        loss, grads = grad_fn(state.params, input_ids, labels, dropout_rng)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss
    
    # ── Training loop ──
    print(f"\n{'='*60}")
    print(f"  STARTING TRAINING — {args.steps} steps")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {jax.devices()[0]}")
    print(f"{'='*60}\n")
    
    n_batches = max(n_samples // args.batch_size, 1)
    epoch = 0
    step_times = []
    loss_history = []
    
    # Checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    for step in range(1, args.steps + 1):
        t0 = time.time()
        
        # Get batch (cyclic through data)
        batch_idx = (step - 1) % n_batches
        start = batch_idx * args.batch_size % n_samples
        end = min(start + args.batch_size, n_samples)
        if end - start < args.batch_size:
            # Wrap around
            diff = args.batch_size - (end - start)
            indices = np.concatenate([
                np.arange(start, end),
                np.arange(diff)
            ])
        else:
            indices = np.arange(start, end)
        
        batch_tokens = tokens_arr[indices]  # [B, seq_len]
        input_ids = jnp.array(batch_tokens, dtype=jnp.int32)
        # Labels: predict next token (shift by 1)
        labels = jnp.concatenate([
            input_ids[:, 1:],
            jnp.zeros((input_ids.shape[0], 1), dtype=jnp.int32)
        ], axis=1)
        
        # Train step
        dropout_rng = jax.random.fold_in(dropout_rng, step)
        state, loss = train_step(state, input_ids, labels, dropout_rng)
        loss_val = float(loss)
        loss_history.append(loss_val)
        step_times.append(time.time() - t0)
        
        # Update epoch counter
        if step % n_batches == 0:
            epoch += 1
        
        # Log
        if step <= 20 or step % 50 == 0:
            avg_time = np.mean(step_times[-50:]) if len(step_times) >= 50 else np.mean(step_times)
            avg_loss = np.mean(loss_history[-50:]) if len(loss_history) >= 50 else loss_val
            lr_val = schedule(step)
            tokens_per_sec = args.batch_size * seq_len / avg_time
            
            print(f"  Step {step:5d}/{args.steps} | "
                  f"loss: {loss_val:.4f} (avg: {avg_loss:.4f}) | "
                  f"lr: {lr_val:.2e} | "
                  f"tok/s: {tokens_per_sec:.0f} | "
                  f"time: {avg_time*1000:.0f}ms")
        
        # Save checkpoint
        if step % 1000 == 0:
            ckpt_path = os.path.join(args.save_dir, f'checkpoint_{step}.msgpack')
            print(f"\n  💾 Saving checkpoint to {ckpt_path}")
            # Save parameters
            params_np = jax.tree.map(lambda x: np.array(x), state.params)
            import pickle
            with open(ckpt_path.replace('.msgpack', '.pkl'), 'wb') as f:
                pickle.dump({'params': params_np, 'config': config, 'step': step, 'loss': loss_val}, f)
    
    # ── Final save ──
    final_path = os.path.join(args.save_dir, 'model_final.pkl')
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE — {args.steps} steps")
    print(f"  Final loss: {np.mean(loss_history[-100:]):.4f}")
    print(f"  Saving to {final_path}")
    
    params_np = jax.tree.map(lambda x: np.array(x), state.params)
    import pickle
    with open(final_path, 'wb') as f:
        pickle.dump({
            'params': params_np,
            'config': config,
            'steps': args.steps,
            'loss_history': loss_history
        }, f)
    
    print(f"\n  ✓ Model saved!")
    print(f"  Loss history: min={min(loss_history):.4f}, final={loss_val:.4f}")


if __name__ == '__main__':
    main()
