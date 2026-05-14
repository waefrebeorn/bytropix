# WuBuText AI — Project Status (May 14, P4 integrated)

## Model
Qwen3.6-35B-A3B from scratch in C + CUDA. 40-layer hybrid (30 SSM + 10 GQA), 2048 hidden, 248K vocab, 256 MoE experts. Poincaré ball hyperbolic geometry (R=0.956).

## Status: Training Pipeline

### ✅ Phase 1-3: Foundation (COMPLETE)
- CPU forward: verified 12.66 CE loss
- GPU forward + save intermediates: verified all 40L
- Hyperbolic GPU kernels: verified (exp_map, log_map, Möbius ops, RSGD optimizer)
- Embedding graft: Poincaré-mapped embeddings on disk
- Training loop: GPU forward + CPU backward, deferred SGD with OpenMP
- Q-learner LR controller: C port
- Gradient clipping: per-element + per-layer norm

### ✅ Phase 4: MoE — WIRED (frozen, dequant-limited)
- `src/wubu_moe_backward.c`: Full MoE backward function (shared expert, routed experts, router softmax gradient)
  - Handles NULL weight pointers gracefully
  - Correctly routes gradients through top-k selection + softmax renormalization
- `gguf_reader.c`: RAM buffer support (`gguf_buffer_data`) — reads entire GGUF file into memory, eliminates SSD seeks
- `train_gpu.c`: MoE backward wired into training loop
  - Gradient chain: MoE → post-norm → attention → pre-norm (all correct)
  - Fixed dangling `d_x_post` (malloc→calloc + residual add)
- **Frozen**: Gradients flow through MoE as identity (expert dequant too slow: 120×268M elements/step)
- To enable: optimize dequantization or pre-load expert weights quantized in RAM

### ⬜ Phase 5: Vision — Not started
- Moondream3 ViT: need weight dumper + C forward encoder

### ⬜ Phase 6: Poincaré Backward — Not started
- Need `gpu_poincare_ssm_forward_save` + `wubu_poincare_ssm_backward`
- Gyration chain rule math in THEORY/

## Performance
| Component | Time |
|-----------|------|
| Model init (40L) | 0.8s |
| GGUF RAM buffer (11 GB) | ~30s |
| GPU forward (40L) | ~2s |
| CPU backward (identity MoE) | ~25s |
| **Total per step** | **~27s** |

## Known Issues
- **loss=nan**: Pre-existing with 35B-A3B model; may need LR/init tuning
- **MoE frozen**: 120 dequantization passes per step too slow
- **No Poincaré backward**: Falls through as identity when POINCARE_R is set

## Architecture (current)
```
Training loop (train_gpu.c):
1. GPU forward through all 40 layers → save intermediates
2. CPU: output projection → CE loss → d_logits
3. CPU backward (deferred grads, reverse order):
   - RMSNorm backward (post-norm)
   - MoE backward (identity, P4 code exists for full)
   - SSM/GQA backward (exact, with weight grads)
   - RMSNorm backward (pre-norm)
4. Batch weight update (OpenMP + deferred cudaMemcpyAsync)
5. Q-learner: loss → Q-table update → new LR
```
