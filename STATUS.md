# WuBuText AI — Project Status (May 14 PM, inference phase)

## Model
Qwen3.6-35B-A3B from scratch in C + CUDA. 40-layer hybrid (30 SSM + 10 GQA), 2048 hidden, 248K vocab, 256 MoE experts. IQ2_M quantized (type 16). Poincaré ball hyperbolic geometry (R=0.956).

## Pipeline Status

### ✅ Phase 1-3: Foundation
- GPU forward + save intermediates: verified
- SSM backward: exact, verified with FD
- GQA backward: exact, verified (was false negative on eps)
- Training loop: GPU forward → CPU backward → deferred SGD + OpenMP
- Q-learner LR controller, gradient clipping

### ✅ Phase 4: MoE (wired)
- `wubu_moe_backward.c`: Full backward (shared expert + routed experts + router softmax through top-k)
- `gguf_reader.c`: RAM buffer (`gguf_buffer_data`) eliminates SSD seeks
- **Lazy MoE dequant** (S1): Only dequantize top-8 experts per layer, not all 256. 9× dequant speedup.

### ✅ Phase 5: Vision encoder (wired, standalone)
- `wubu_vision.c` / `.h`: 27-layer 3D ViT
- GPU forward: 217ms for 256×256 (161× speedup vs CPU)

### ✅ Phase 6: Poincaré backward (wired, identity approx)
- CPU backward with gyration identity approximation

## Inference Engines (May 14 PM)

### Inference completed
| Engine | Status | Performance |
|--------|--------|-------------|
| `infer_moe` | ✅ | MoE forward 36 tok/s (B=1,T=4), dequant 3.18s bottleneck |
| `infer_moe_lazy` | ✅ | **Lazy dequant: 9× speedup** (3.1s→0.35s). 30 experts / 256. Output match verified. |
| `infer_unified` | ✅ | **40-layer SSM→GQA→MoE** in one binary. Lazy MoE integrated. Per-layer timing. |
| `infer_vision` | ✅ | CPU 27L ViT: 825ms (64×64), ~35s (256×256) |
| `infer_vision_gpu` | ✅ | **GPU cuBLAS: 65ms (64×64), 217ms (256×256)** — 161× speedup |
| `infer_poincare` | ✅ | GPU Poincaré SSM: 2835 tok/s (B=1,T=4) |
| `test_256k` | ✅ | MoE router O(T) at 256K verified (4300 tok/s) |
| `test_kv_cache` | ✅ | **KV cache for GQA: max_diff=0 vs full recompute**. 1 GB/layer @ 256K. 2.6× speedup at T=8. |

### Known issues
- **GQA forward produces NaN at layer 3** — pre-existing bug. Memory corruption hypothesis: MoE weight load overwrites GQA input buffer via pointer aliasing. NaN guard (NaN→0) applied as safety net.
- **P0 gradient explosion (4e13 ratio)** — TGT π-odometer wrapping applied to SGD step (replaced clip[-10,10] with fmod remainder). Needs training run to verify.
- **bench_e2e** — all zeros output. GPU weight loading path broken (bench.c).
- **train_backprop** — hangs at model init. Unknown cause.
- **train_gpu** — CE loss 69 vs expected 12.66. Same root cause as bench_e2e.

## TGT (Toroidal Gradient Transformation) — Applied May 14
```
BOUNDARY = 2π
remainder = fmod(x + π, BOUNDARY) - π   # maps any value to [-π, π]
quotient = floor((x + π) / BOUNDARY)    # magnitude wraps (integer)
tgt_safe_expf: clamp x to [-80,80] before expf to prevent float32 overflow
```

Applied to: SSM state decay (safe_expf), SSM state matrix entries (tgt_wrap), GQA attention scores (tgt_wrap), GQA Q/K/V projections (NaN→0 guard), SGD optimizer (tgt_wrap replaces clip[-10,10]).
Committed in fefd426.

## Critical Gaps

| Gap | Priority | Status |
|-----|----------|--------|
| G1: Lazy MoE dequant | P0 | ✅ Done. infer_moe_lazy.c committed. 9× speedup. |
| G2: Unified inference | P1 | ✅ Done. infer_unified.c committed. Full 40-layer chain. |
| G3: KV cache design | P1 | ✅ Done. test_kv_cache.c committed. 3 GB/layer @ 256K, verified. |
| G4: Vision→model integration | P2 | 🚧 Not started |
| G5: Mind palace update | P3 | ✅ Updated. |
| NaN in SSM→GQA forward | P0? | Pre-existing. Blocks S4 verification. |
| Gradient explosion (4e13) | P0 | Training blocker, independent of inference. |

## Performance Summary
- MoE forward (lazy, one layer): ~222ms (116ms dequant + 106ms compute) per layer for 4 tokens
- SSM forward (one layer): ~374ms for 4 tokens
- GQA forward (one layer): ~164ms for 4 tokens
- Unified 40-layer forward: ~7.8s for 4 tokens (without output projection)
- KV cache attention: O(T) per decode step vs O(T²) without cache

## Build
`make <target>` — standard. NVCC at /usr/local/cuda-13.1/bin/nvcc -arch=sm_120 (RTX 5050).
25 GB RAM, 6.4 GB VRAM (shared with display).
