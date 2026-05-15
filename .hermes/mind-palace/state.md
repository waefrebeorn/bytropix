# WuBuText AI — State Dashboard (May 15 PM v6)

## Inference Engines

| Binary | Status | Performance | Notes |
|--------|--------|-------------|-------|
| `train_integrated` | ✅ | **11s/step**, CE 21.6→18.4, 0 NaN | Primary — per-expert dequant, GPU proj |
| `infer_moe_lazy` | ✅ | 37 tok/s, 0.35s dequant (9×) | Lazy dequant: top-8/256 experts |
| `infer_unified` | ✅ | 40 layers in 1 binary | SSM→GQA→MoE chain |
| `test_kv_cache` | ✅ | max_diff=0.00 vs recompute | KV cache: 1GB/layer @ 256K |
| `infer_vision_gpu` | ✅ | GPU: 99ms (128×128) | 27 GPU layers, 0 NaN |
| `infer_poincare` | ✅ | GPU: 2835 tok/s | Poincaré SSM on GPU |
| `test_moe` | ✅ | range [-0.028, 0.031], NaN=0 | 36.6 tok/s |
| `train_gpu` | ✅ | CE=12.42 with lazy MoE | GPU forward + lazy MoE |
| `train_backprop` | ✅ | Runs (CPU-slow ~25s/step) | Not hanging |
| `bench_e2e` | ✅ | GPU weight loading fixed | Match verified |
| `infer_vision_text` | ✅ | Vision→text pipeline | Real screenshot, 0 NaN |

## Cold Gaps — ALL CLOSED (May 14 session)

| Gap | Status | Test Result |
|-----|--------|-------------|
| Poincaré GQA backward | ✅ | dQ=1.95, dK=0.004, dV=0.70, dX=571 |
| Nested SSM backward | ✅ | K=1,2,3 all pass, 0 NaN |
| Möbius linear layer (M⊗) | ✅ | fwd+bwd, ball constraint satisfied |
| Gyration closed-form | ✅ | exact match, ~3× faster |
| Hyperbolic output projection | ✅ | 5/5 pass |
| Nested MoE 2-level backward | ✅ | 15/15 tests pass |
| Hyperbolic KV cache | ✅ | prefill + incremental verified |

## NaN/Inf Fixes

| Location | Fix | Effect |
|----------|-----|--------|
| **gguf_raw_size(IQ2_XXS)** | 72→66 bytes/block (was wrong stride) | Per-expert offset correct |
| **MoE full dequant (3GB/step)** | Per-expert dequant + transpose (3.9ms/expert) | 177s→11s/step (16×) |
| **MoE weight strided extraction** | Per-expert raw offset + correct transpose | Hidden max=13 (was 5e9) |
| SSM state decay | `tgt_safe_expf` clamp [-80,80] in GPU kernels | No exp overflow in 4 sites |
| CPU forward logits | Output projection from output.weight | NaN 0.57%→0% |
| CPU RMSNorm OOB | 6 call sites in test_kv_cache.c patched | No OOB access |

## Paper Audit Discrepancies (32 Qwen files, May 15)

| Parameter | Config Value | C Header | Status |
|-----------|-------------|-----------|--------|
| Full attn head_dim | 256 | `GQA_HEAD_DIM=256` | ✅ Match |
| Linear attn head_dim | 128 | `SSM_D_STATE=128` | ✅ Match |
| GQA KV heads | 2 (8:1 ratio) | `GQA_KV_HEADS=2` | ✅ Match |
| SSM K/V heads | 16 K, 32 V | `SSM_K_HEADS=16, SSM_V_HEADS=32` | ✅ Match |
| Conv kernel | 4 | `CONV_KERNEL=4` | ✅ Match |
| Conv dim | 1536 (from earlier paper doc) | CONV_DIM=8192 = Q(2048)+K(2048)+V(4096) | ✅ **Not a bug** — CONV_DIM is the QKV projection output, not linear_conv_dim |
| MoE experts | 256 | `N_EXPERTS=256` | ✅ Match |
| Active experts | 8 | `N_ACTIVE_EXPTS=8` | ✅ Match |
| Expert FFN dim | 512 | `D_FF=512` | ✅ Match |
| RoPE theta | 10,000,000 | `ROPE_THETA=10000000.0f` (new May 15) | ✅ **Fixed** — added RoPE impl to GPU GQA forward |
| Partial RoPE | 0.25 (64/256) | `ROTARY_DIM=64` (new May 15) | ✅ **Fixed** — apply_rotary to first 64 dims |
| MRoPE 3D | section=[11,11,10] | ✅ Equivalent for text-only (t=h=w=seq_pos) | Full 3D blocked on vision interleaving P3 |
| MTP head | 1 layer | ✅ **Fixed May 15** — auxiliary t+2 loss with gradient, weight=0.3, reuse output_weight | Training-only, no dedicated MTP weights in GGUF |
| bos/eos | both 248044 | Tokenizer reads GGUF | ✅ Match |
| rms_norm_eps | 1e-06 | 1e-6f in code | ✅ Match |
| router_aux_loss_coef | 0.001 | (optional, no explicit impl) | 🔍 Verify |

## Tailslayer Findings (May 15)

| Pattern | WuBuText Analog | Priority |
|---------|----------------|----------|
| N replicas on independent DRAM channels | N draft tokens speculated in parallel | **P2** |
| clflush+reload timing | Forward pass timing for draft verification | P2 |
| Hedged read (first-response-wins) | Accept longest valid prefix, cancel remaining | **P2** |
| Sliding window pair sampling | Draft-target logit time alignment | P2 |
| tREFI probe (TSC calibration, harmonic binning) | CUDA kernel launch / PCIe timing | P3 |
| N replicas pinned → separate cores | E experts dispatched → S SMs | P3 |
| Physical addr → channel bit extraction | CUDA shared memory bank conflict analysis | P3 |

## Known Issues

| Issue | Impact | Status |
|-------|--------|--------|
| ~11s/step GPU compute (40 layers) | Training slow vs GPU-only models | GPU MoE forward or double-buffering could reduce |
| PGA loss jumps 21.6→69 | PGA backward LR too high | Needs LR scaling investigation |
| CPU output projection ~2s/token | O(N*V*D) bottleneck for V=248320 | Move to GPU |
| CONV_DIM=8192 vs config 1536 | Possible off-by-one in SSM layernorm/conv | Needs code audit |
| MRoPE 3D not implemented | Position encoding degrades >32K | P2 |
