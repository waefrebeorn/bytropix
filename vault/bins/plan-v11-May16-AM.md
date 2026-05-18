# bytropix — Plan (May 18 — Phase 2 Complete)

## Purpose
Achieve 1:1 inference parity with llama.cpp + serve Qwen3.6-35B-A3B-UD-IQ2_M as a real inference engine.

## Completed This Week

| Fix | Impact | Evidence |
|-----|--------|----------|
| GQA Q/gate interleave | cos-sim -0.51 → 0.9968 | test_full_moe |
| IMRoPE | Multi-token position encoding | T=2 test passes |
| MoE OpenMP | 3× speedup (44ms→15ms/layer) | PROFILE=1 |
| Buffer reuse | 160→5 mallocs/forward | Code inspection |
| gen_text pipeline | Coherent 32-token generation | "The capital of France is Paris" |
| ref_dumper tool | Direct libllama.so reference | Per-layer comparison |
| vault papers read | Qwen arch, Unsloth UD, DA v10 | Arch understanding complete |

## Remaining Work

### P0 — Chat Template (DA Gap 7)
Apply Qwen chat template before gen_text tokenization.
Adds system/assistant role markers for better quality.

### P0 — Multi-Token Cos-Sim Verification
T=1 verified at 0.9968. T>1 NOT verified against llama.cpp.
Run ref_dumper with multi-token prompt, compare each step.

### P1 — KV Cache for GQA Decode
Avoid full-attention recompute per decode step.
Impact: ~10% decode speedup.

### P1 — SIMD vec_dot
Current generic C dot product causes 0.003 cos-sim gap.
Replace with SSE2/AVX2 intrinsics for cos-sim → 1.0.

### P2 — GPU Decode Path
Wire existing GPU kernels (gpu_gqa_forward, gpu_ssm_forward) into gen_text loop.
Requires solving PCIe data transfer bottleneck for MoE weights.

## 256K Context Roadmap

| Step | What | Status |
|------|------|--------|
| 1 | GQA KV cache (append-only) | ✅ GPU kernels exist |
| 2 | SSM state carry | ✅ Verified |
| 3 | Lazy MoE cache | ✅ Implemented |
| 4 | GPU forward for GQA/SSM decode | ✅ Kernels exist (not wired) |
| 5 | Verify 256K forward pass | ⬜ Not yet |
| 6 | Single-token generation at 256K | ⬜ Not yet |
| 7 | Tailslayer spec decode | ⬜ Not yet |

## Verification Protocol

```
# Quick cos-sim check
make test_full_moe && ./test_full_moe
# Expected: cos-sim 0.9968

# Generation check
make gen_text && ./gen_text "Hello" 8
# Expected: coherent English text

# Per-layer profile
PROFILE=1 ./test_full_moe
# Expected: MoE ~15ms, SSM ~13ms, Output ~12ms
```
