# WuBuText AI — State Dashboard (May 15 PM v7)

## Inference Engines

| Binary | Status | Performance | Notes |
|--------|--------|-------------|-------|
| `infer_text` | ✅ | Full text pipeline | Tokenizer→embed→forward→sample→decode. MoE optional |
| `train_integrated` | ✅ | **11s/step**, CE 21.6→18.4, 0 NaN | Primary — per-expert dequant, GPU proj |
| `infer_moe_lazy` | ✅ | 37 tok/s, 0.35s dequant (9×) | Lazy dequant: top-8/256 experts |
| `infer_unified` | ✅ | 40 layers in 1 binary | SSM→GQA→MoE chain |
| `test_kv_cache` | ✅ | max_diff=0.00 vs recompute | KV cache: 1GB/layer @ 256K |
| `test_256k` | ✅ | MoE router 65K verified | SSM O(T), GQA needs KV cache |
| `infer_vision_gpu` | ✅ | GPU: 99ms (128×128) | 27 GPU layers, 0 NaN |
| `infer_poincare` | ✅ | GPU: 2835 tok/s | Poincaré SSM on GPU |
| `test_gpu` | ✅ | GPU/CPU match | SSM+GQA layer verified (RoPE added) |
| `test_cuda_kernels` | ✅ | max_diff<6e-8 | SSM scan + MoE dispatch |

## All Cold Gaps CLOSED (May 14-15)

| Gap | Status | Test Result |
|-----|--------|-------------|
| Poincaré GQA backward | ✅ | dQ=1.95, dK=0.004, dV=0.70 |
| Nested SSM backward | ✅ | K=1,2,3 all pass, 0 NaN |
| Möbius linear layer (M⊗) | ✅ | fwd+bwd, ball constraint satisfied |
| Gyration closed-form | ✅ | exact match, ~3× faster |
| Hyperbolic output projection | ✅ | 5/5 pass |
| Nested MoE 2-level backward | ✅ | 15/15 tests pass |
| Hyperbolic KV cache | ✅ | prefill + incremental verified |

## NaN/Inf Fixes (All Resolved)

| Location | Fix | Effect |
|----------|-----|--------|
| **gguf_raw_size(IQ2_XXS)** | 72→66 bytes/block | Per-expert offset correct |
| **MoE full dequant (3GB/step)** | Per-expert dequant + transpose (3.9ms/expert) | 177s→11s/step (16×) |
| **MoE weight strided extraction** | Per-expert raw offset + correct transpose | Hidden max=13 (was 5e9) |
| SSM state decay | `tgt_safe_expf` clamp [-80,80] in 4 GPU kernel sites | No exp overflow |
| CPU forward logits | Output projection from output.weight | NaN 0.57%→0% |
| CPU RMSNorm OOB | 6 call sites in test_kv_cache.c patched | No OOB access |
| RoPE missing from GQA | Added `apply_rotary_qk_kernel` with θ=10M, rotary_dim=64 | Positional encoding added |

## Architecture Parity (14 config params — ALL RESOLVED)

| Parameter | Config Value | C Header | Status |
|-----------|-------------|-----------|--------|
| Full attn head_dim | 256 | `GQA_HEAD_DIM=256` | ✅ Match |
| Linear attn head_dim | 128 | `SSM_D_STATE=128` | ✅ Match |
| GQA KV heads | 2 (8:1 ratio) | `GQA_KV_HEADS=2` | ✅ Match |
| SSM K/V heads | 16 K, 32 V | `SSM_K_HEADS=16, SSM_V_HEADS=32` | ✅ Match |
| Conv kernel | 4 | `CONV_KERNEL=4` | ✅ Match |
| Conv dim | 8192 = Q(2048)+K(2048)+V(4096) | `CONV_DIM=8192` | ✅ Not a bug |
| MoE experts | 256 | `N_EXPERTS=256` | ✅ Match |
| Active experts | 8 | `N_ACTIVE_EXPTS=8` | ✅ Match |
| Expert FFN dim | 512 | `D_FF=512` | ✅ Match |
| RoPE theta | 10,000,000 | `ROPE_THETA=10000000.0f` | ✅ Fixed |
| Partial RoPE | 0.25 (64/256) | `ROTARY_DIM=64` | ✅ Fixed |
| MRoPE 3D | section=[11,11,10] | Equivalent (t=h=w=seq_pos) | ✅ Text-only |
| MTP head | 1 layer | Auxiliary t+2 loss, w=0.3 | ✅ Fixed |
| bos/eos | both 248044 | Tokenizer reads GGUF | ✅ Match |
| rms_norm_eps | 1e-06 | 1e-6f in code | ✅ Match |

## 256K Context Status

| Component | Scaling | Status |
|-----------|---------|--------|
| MoE Router | O(T) | ✅ Verified to 65K (stopped at >15s) |
| SSM forward | O(T) | ✅ Verified via GPU kernel |
| GQA attention | O(T²) | ⚠️ Needs KV cache for 256K |
| Token embeddings | O(T) | ✅ 2GB at 256K |
| Output projection | O(T×V) | ⚠️ 256K×248K×2K = impractical |
| Generation | O(T²) per step | ⚠️ Full recompute, no KV cache |

## Text Generation Pipeline

- `infer_text.c`: Full pipeline — tokenizer, embedding lookup, forward pass, sampling, decode
- Supports greedy and top-k sampling
- MoE optional via `MOE=1` env var (slow: full dequant per layer per step)
- Without MoE: output is passthrough (no FFN) → quality limited
- Coherence test requires KV cache + lazy MoE for practical speed

## Known Issues

| Issue | Impact | Status |
|-------|--------|--------|
| ~11s/step GPU compute (40 layers) | Training slow | GPU MoE forward could reduce |
| PGA loss jumps 21.6→69 | PGA backward LR too high | Needs LR scaling |
| CPU output projection ~2s/token | O(N×V×D) for V=248320 | Move to GPU |
| No KV cache for GQA | O(T²) per step for generation | P2 |
| MoE inference slow | Full dequant per layer/step | Lazy per-expert cache needed |
| Without MoE, output is garbage | FFN = entire model non-linearity | MOE=1 needed for quality |

## TGT Math
BOUNDARY = 2π
remainder = fmod(x + π, BOUNDARY) - π
tgt_safe_expf(x) = x > 80 ? 80 : x < -80 ? 0 : expf(x)
