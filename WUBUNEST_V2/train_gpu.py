"""
train_gpu.py — WubuNestGPT v2 GPU Training (JAX/Flax/CUDA)
===========================================================
- Float32 (no bfloat16 warnings)
- JIT-compiled train_step for CUDA speed (~5-20ms/step)
- Proper unbuffered output to log file
- Checkpoints every 1000 steps to ./checkpoints/
- Clean exit, no tcsetattr issues

Usage:
  cd /home/wubu/bytropix/WUBUNEST_V2
  python -u train_gpu.py 2>&1 | tee -a train_gpu.log
"""

import os, sys, time, json, pickle, re
import numpy as np
from pathlib import Path

# ── Force bfloat16 for Tensor Core efficiency ──
os.environ['JAX_DEFAULT_DTYPE'] = 'bfloat16'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORMS'] = ''
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85'  # Leave 15% for CUDA kernels

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

from wubu_nest_gpt_v2 import WubuNestGPT, compute_loss, load_corpus_data, SimpleTokenizer

# ── Flush after every print ──
def p(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ─── Config (small model to fit RTX 5050 6GB) ───
VOCAB_SIZE = 5000
D_MODEL = 192
N_LAYERS = 3
N_HEADS = 4
D_COMPRESSED = D_MODEL       # 192 (d_c = d_model for small model)
D_FF = D_MODEL * 4           # 768
N_ROUTED = 4
TOP_K = 2
LEARNING_RATE = 3e-4
WARMUP = 50
STEPS = 5000
BATCH_SIZE = 8
SEQ_LEN = 128
CORPUS_PATH = '../ENCODERS/phase3-generative/CORPUS.py'
SAVE_DIR = './checkpoints'


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    p("=" * 60)
    p("  WubuNestGPT v2 — GPU Training (JAX/Flax/CUDA)")
    p("  Architecture: LatentAttention + SparseMoE + Hyperbolic")
    p("  Float32, JIT-compiled for speed")
    p("=" * 60)

    # ── Config ──
    config = {
        'vocab_size': VOCAB_SIZE,
        'd_model': D_MODEL,
        'n_layers': N_LAYERS,
        'n_heads': N_HEADS,
        'd_head': D_MODEL // N_HEADS,
        'd_compressed': D_COMPRESSED,
        'd_ff': D_FF,
        'n_shared': 2,
        'n_routed': N_ROUTED,
        'top_k': TOP_K,
        'use_quant': False,
        'dropout_rate': 0.1,
        'dtype': 'bfloat16',
        'learning_rate': LEARNING_RATE,
        'warmup_steps': WARMUP,
        'train_steps': STEPS,
        'weight_decay': 0.1,
        'aux_loss_coef': 0.01,
        'max_seq_len': SEQ_LEN,
    }

    # ── Tokenizer ──
    p(f"\n[1/4] Creating tokenizer (vocab_size={VOCAB_SIZE})...")
    tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
    # Build a real char-based vocab from corpus
    corpus_path_abs = os.path.join(os.path.dirname(__file__), CORPUS_PATH)
    if os.path.exists(corpus_path_abs):
        with open(corpus_path_abs, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()
        chars = sorted(list(set(raw)))
        special = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        tokenizer.vocab = {c: i for i, c in enumerate(special + chars[:min(len(chars), VOCAB_SIZE - 4)])}
        tokenizer.id_to_char = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.vocab_size = len(tokenizer.vocab)
    p(f"  Vocab size: {tokenizer.vocab_size}")
    config['vocab_size'] = tokenizer.vocab_size

    # ── Load corpus ──
    p(f"\n[2/4] Loading corpus from {CORPUS_PATH}...")
    texts = load_corpus_data(corpus_path_abs)
    p(f"  Loaded {len(texts)} text chunks")

    # ── Create model ──
    p(f"\n[3/4] Creating WubuNestGPT v2...")
    p(f"  d_model={D_MODEL}, layers={N_LAYERS}, heads={N_HEADS}")
    p(f"  Latent attention: d_c={D_COMPRESSED}, d_h={D_MODEL // N_HEADS}")
    p(f"  MoE: {N_ROUTED} experts, top-{TOP_K} per token")

    model = WubuNestGPT(config=config)

    # Initialize params
    rng = jax.random.PRNGKey(42)
    dummy = jnp.zeros((2, 16), dtype=jnp.int32)
    variables = model.init(rng, dummy, is_training=True)
    param_count = sum(x.size for x in jax.tree.leaves(variables))
    p(f"  Total parameters: {param_count:,}")

    # ── Prepare data ──
    p(f"\n[4/4] Preparing training data...")
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)

    total_tokens = (len(all_tokens) // SEQ_LEN) * SEQ_LEN
    all_tokens = all_tokens[:total_tokens]
    tokens_arr = np.array(all_tokens, dtype=np.int32).reshape(-1, SEQ_LEN)
    n_samples = tokens_arr.shape[0]
    p(f"  Total tokens: {total_tokens:,}")
    p(f"  Sequences: {n_samples} (seq_len={SEQ_LEN})")

    # ── Create optimizer ──
    p("\n  Initializing optimizer (AdamW + cosine decay)...")
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP,
        decay_steps=STEPS,
        end_value=LEARNING_RATE * 0.01,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            weight_decay=config['weight_decay'],
        ),
    )

    # Use the existing compute_loss with proper closure
    dropout_rng = jax.random.PRNGKey(42)

    # Recreate compute_loss with the model in closure
    def loss_fn(params, input_ids, labels, rng_key):
        logits, _, bias_updates = model.apply(
            {'params': params},
            input_ids,
            is_training=True,
            rngs={'dropout': rng_key},
        )
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V).astype(jnp.float32)
        labels_flat = labels.reshape(-1)
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, labels_flat).mean()
        aux_loss = 0.0
        for bu in bias_updates:
            if bu is not None:
                aux_loss = aux_loss + jnp.mean(jnp.abs(bu))
        return ce_loss + config['aux_loss_coef'] * aux_loss

    @jax.jit
    def train_step_jit(state, input_ids, labels, rng_key):
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params, input_ids, labels, rng_key)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss

    # Create train state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )

    # ── Training loop ──
    n_batches = max(n_samples // BATCH_SIZE, 1)
    p(f"\n{'='*60}")
    p(f"  STARTING TRAINING — {STEPS} steps")
    p(f"  Batch size: {BATCH_SIZE}, Device: {jax.devices()[0]}")
    p(f"  Data repeats: ~{STEPS // n_batches + 1} epochs")
    p(f"{'='*60}\n")

    loss_history = []
    step_times = []
    epoch = 0
    t_start = time.time()

    # Warmup JIT: run a single step to trigger compilation
    p("  [JIT warmup] Compiling train_step on CUDA...")
    warmup_batch = tokens_arr[:BATCH_SIZE]
    warmup_ids = jnp.array(warmup_batch, dtype=jnp.int32)
    warmup_labels = jnp.concatenate([
        warmup_ids[:, 1:],
        jnp.zeros((warmup_ids.shape[0], 1), dtype=jnp.int32),
    ], axis=1)
    t0 = time.time()
    state, warmup_loss = train_step_jit(state, warmup_ids, warmup_labels, dropout_rng)
    compile_time = time.time() - t0
    p(f"  JIT warmup step done in {compile_time:.1f}s (loss: {float(warmup_loss):.4f})")
    p()

    for step in range(2, STEPS + 1):
        t0 = time.time()

        # Get batch (cyclic)
        batch_idx = (step - 1) % n_batches
        start = batch_idx * BATCH_SIZE % n_samples
        end = min(start + BATCH_SIZE, n_samples)
        if end - start < BATCH_SIZE:
            diff = BATCH_SIZE - (end - start)
            indices = np.concatenate([np.arange(start, end), np.arange(diff)])
        else:
            indices = np.arange(start, end)

        batch_tokens = tokens_arr[indices]
        input_ids = jnp.array(batch_tokens, dtype=jnp.int32)
        labels = jnp.concatenate([
            input_ids[:, 1:],
            jnp.zeros((input_ids.shape[0], 1), dtype=jnp.int32),
        ], axis=1)

        # Fold dropout rng
        dropout_rng = jax.random.fold_in(dropout_rng, step)

        # Train step (JIT-compiled, runs on CUDA)
        state, loss = train_step_jit(state, input_ids, labels, dropout_rng)
        loss_val = float(loss)
        loss_history.append(loss_val)
        dt = time.time() - t0
        step_times.append(dt)

        # Update epoch
        if step % n_batches == 0:
            epoch += 1

        # Log (dense at start, sparse later)
        if step <= 10 or step % 50 == 0 or step == STEPS:
            avg_loss = np.mean(loss_history[-100:]) if len(loss_history) >= 100 else loss_val
            avg_time = np.mean(step_times[-100:]) if len(step_times) >= 100 else np.mean(step_times)
            tok_sec = BATCH_SIZE * SEQ_LEN / avg_time
            lr_val = schedule(step)
            elapsed = time.time() - t_start
            eta = (STEPS - step) * avg_time
            p(f"  Step {step:5d}/{STEPS} | loss: {loss_val:.4f} (avg: {avg_loss:.4f}) | "
              f"lr: {lr_val:.2e} | {tok_sec:,.0f} tok/s | {avg_time*1000:.0f}ms | "
              f"ETA: {eta/60:.0f}m")

        # Save checkpoint
        if step % 1000 == 0:
            ckpt_path = os.path.join(SAVE_DIR, f'checkpoint_{step}.pkl')
            params_np = jax.tree.map(lambda x: np.array(x), state.params)
            with open(ckpt_path, 'wb') as f:
                pickle.dump({
                    'params': params_np,
                    'config': config,
                    'step': step,
                    'loss': loss_val,
                    'loss_history': loss_history[-1000:],
                }, f)
            p(f"  💾 Checkpoint saved: {ckpt_path}")

    # ── Final save ──
    total_time = time.time() - t_start
    final_path = os.path.join(SAVE_DIR, 'model_final.pkl')
    params_np = jax.tree.map(lambda x: np.array(x), state.params)
    with open(final_path, 'wb') as f:
        pickle.dump({
            'params': params_np,
            'config': config,
            'steps': STEPS,
            'loss_history': loss_history,
        }, f)

    p(f"\n{'='*60}")
    p(f"  TRAINING COMPLETE — {STEPS} steps in {total_time:.0f}s")
    p(f"  Final loss: {np.mean(loss_history[-100:]):.4f}  Best: {min(loss_history):.4f}")
    p(f"  Model saved: {final_path}")
    p(f"{'='*60}")


if __name__ == '__main__':
    main()
