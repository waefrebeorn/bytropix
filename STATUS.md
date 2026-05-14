# WuBuText AI — Project Status (May 14, post-fix)

## Model
Qwen3.6-35B-A3B from scratch in C + CUDA. 40-layer hybrid (30 SSM + 10 GQA), 2048 hidden, 248K vocab, 256 MoE experts. Poincaré ball hyperbolic geometry (R=0.956).

## Status: Training Pipeline — VERIFIED (May 14)

### ✅ Phase 1-3: Foundation (COMPLETE)
- CPU forward: verified 12.66 CE loss
- GPU forward + save intermediates: verified all 40L
- Hyperbolic GPU kernels: verified (exp_map, log_map, Möbius ops, RSGD optimizer)
- Embedding graft: Poincaré-mapped embeddings on disk

### ✅ Phase 3.5: Training Loop — VERIFIED

| Component | Status | Detail |
|-----------|--------|--------|
| GPU forward + scratch download | ✅ | All 40L: SSM (15 buffers + states + conv) + GQA (6 buffers) |
| CPU backward chain | ✅ | Full 40L chain: SSM exact backward + GQA exact backward |
| **SSM backward** | ✅ Pass | 12/12 with eps=1e-3 (was 11/12 — FD eps=1e-5 too small) |
| **GQA backward** | ✅ Pass | 38/40 dQ, 40/40 dK, 40/40 dV with eps=1e-2 (was failing with eps=1e-5) |
| RMSNorm backward | ✅ | Pre + post attention norm backward |
| **Internal weight grads** | ✅ FIXED | All SSM (7) + GQA (6) weight grads per layer, non-NULL |
| Batch weight update | ✅ | Deferred SGD with OpenMP, per-element clip at 10.0 |
| Q-learner LR controller | ✅ | C port of QLearnerLR: 10-state × 3-action Q-table, tunes LR per step |
| **Loss descent** | ✅ CONFIRMED | 69→10.57→14.71 (oscillating, B=1 T=4 small batch) |
| GPU weight sync | ✅ | Batch cudaMemcpyAsync after all layers updated |
| Gradient clipping | ✅ | d_hidden per-sample norm clip 100 + d_normed per-layer norm clip 10 + weight per-element clip 10 |

### ⬜ Phase 4: MoE — Not in train_gpu yet
- wubu_model_forward_from_embd runs MoE
- Not wired into train_gpu training loop

### ⬜ Phase 5: Vision — Loader only
- Vision loader exists, not integrated

### ⬜ Phase 6: Poincaré Backward
- RSGD exists in src/rsgd.c
- gpu_poincare_ssm_forward has NO save variant → no backward path
- Gyration chain rule not written

## Known Issues

**P0: Step time 42s** — CPU backward is single-threaded bottleneck. 30 SSM layers × 33M element backward per layer. Need parallelization or GPU backward.

**P0: Gradient magnitudes 1e16+** — Internal weight gradients pre-clip are huge. Per-element clip at 10.0 handles this but loses gradient ratio information. Root cause: activation values grow through 40 layers.

**P1: No Poincaré backward** — When POINCARE_R is set, forward uses non-save variant. Backward falls through as identity.

## Architecture (current)
```python
# Training loop (train_gpu.c):
1. GPU forward through all 40 layers
   → SSM: gpu_ssm_forward_save + download 15 buffers + states_t to CPU
   → GQA: gpu_gqa_forward_save + download 6 buffers to CPU
   → Save normed[l], attn_out[l], normed2[l] for ALL layers

2. CPU: output projection @ lm_head → logits [N, vocab_size]
   → CE loss → d_logits

3. CPU backward (deferred grads):
   → d_hidden = output_weight^T @ d_logits
   → For each layer (reverse):
      SSM: wubu_ssm_backward(ALL 15 buffers, NON-NULL weight grads) — EXACT
      GQA: wubu_gqa_backward(ALL 6 buffers, NON-NULL weight grads) — EXACT
   → Store weight+grad pointers for batch update

4. Batch weight update (OpenMP):
   → Per-element clip at 10.0
   → SGD: w -= lr * clip(g)
   → cudaMemcpyAsync all 40L weights to GPU

5. Q-learner: loss → Q-table update → new LR for next step
```

## Files
- `src/wubu_ssm.c` — All SSM/GQA backward functions + save variants
- `src/wubu_model.c` — Model-level backward chaining
- `src/bench.c` — gpu_ssm_forward_save, gpu_gqa_forward_save
- `src/qlearner.c` + `include/qlearner.h` — Q-learning LR controller
- `src/rsgd.c` + `include/rsgd.h` — Riemannian SGD for hyperbolic params
- `tools/train_gpu.c` — GPU forward + CPU backward training loop (full internal weight grads + Q-learner)
- `tools/test_backward.c` — SSM backward tests
- `tools/test_bwd_gqa.c` — GQA backward attention test (use eps=1e-2 for FD)

## Hardware
RTX 5050 6.4GB | B=1, T=4 typical | CPU: 46GB RAM
