# WuBuText AI — Project Status (May 14, all PE phases wired)

## Model
Qwen3.6-35B-A3B from scratch in C + CUDA. 40-layer hybrid (30 SSM + 10 GQA), 2048 hidden, 248K vocab, 256 MoE experts. Poincaré ball hyperbolic geometry (R=0.956).

## Pipeline Status

### ✅ Phase 1-3: Foundation
- GPU forward + save intermediates: verified
- SSM backward: exact, verified with FD
- GQA backward: exact, verified (was false negative on eps)
- Training loop: GPU forward → CPU backward → deferred SGD + OpenMP
- Q-learner LR controller, gradient clipping

### ✅ Phase 4: MoE (wired, frozen)
- `wubu_moe_backward.c`: Full backward (shared expert + routed experts + router softmax through top-k)
- `gguf_reader.c`: RAM buffer (`gguf_buffer_data`) eliminates SSD seeks
- `train_gpu.c`: MoE wired into gradient chain (post-norm → MoE → attn → pre-norm)
- **Frozen**: 120 expert dequant ops/step too slow. Identity gradient fallback.

### ✅ Phase 5: Vision encoder (wired, untested)
- `wubu_vision.c` / `.h`: 27-layer 3D ViT (temporal_patch=2, spatial_merge=2)
- Patch embedding, position embeddings, GQA attention, GELU, LayerNorm
- Merger projection (mm.0 → mm.2 → 2048 dim)
- Works for any image size (auto-computes patch count)

### ✅ Phase 6: Poincaré backward (wired, identity approx)
- `cuda_kernels.cu`: Poincaré recurrence saves state trajectory (d_states_t)
- `bench.c`: `gpu_poincare_ssm_forward_save` captures all timesteps
- `wubu_poincare_ssm_backward.c`: CPU backward (steps 10-12, 1-8 correct; step 9 gyration identity approximation)

## Performance Bottlenecks

| Bottleneck | Detail |
|------------|--------|
| MoE dequant | 120 × 268M elements per step |
| Poincaré gyration | Step 9 backward needs Möbius chain rule |
| Vision integration | Not wired to training (standalone encoder) |

## Inference Engines (May 14)

| Engine | Status | Performance |
|--------|--------|-------------|
| `infer_moe` | ✅ | MoE forward 36 tok/s (B=1,T=4), dequant 3.18s bottleneck |
| `infer_vision` | ✅ | CPU 27L ViT: 825ms (64×64), ~35s (256×256) — OpenMP enabled |
| `infer_vision_gpu` | ✅ | **GPU cuBLAS ViT: 65ms (64×64), 217ms (256×256)** — 161× speedup |
| `infer_poincare` | ✅ | GPU Poincaré SSM: 2975 tok/s (B=1,T=4) |
| `test_256k` | ✅ | MoE router verified O(T) at 256K (4500 tok/s), SSM extrapolated |
| MoE lazy dequant | 🚧 | Only dequantize top-2 experts per token |
| Vision GPU | 🚧 | cuBLAS for linear layers → target <1s |

## Next: Optimize Everything

All PE phases wired. Inference engines built. Ready for optimization pass:
- Fast dequant: pre-load quantized weights, SIMD dequant
- MoE: batch dequant, once per epoch  
- Poincaré: full gyration backward from math in THEORY/
- Vision: GPU forward with cuBLAS
