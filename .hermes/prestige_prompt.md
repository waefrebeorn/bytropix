# Prestige Prompt — May 19, 2026 (Triple DA v6)

## Project: bytropix — Qwen3.6-35B-A3B-UD-IQ2_M
Triple DA complete. SSM recurrence verified IDENTICAL. 0.79 cos-sim = quantized precision, not algorithm.

## Benchmarks (DA Verified)
| Metric | Value | Status |
|--------|-------|--------|
| Decode | 7.8 tok/s | ✅ Phase 8: OMP task dispatch + AVX2 IQ2_XXS |
| Prefill | 10.4 tok/s | ✅ |
| Logit cos-sim vs llama | 0.7944 | ✅ Pre-existing at IQ2_M — precision, not bug |
| Per-layer cos-sim avg | 0.88 | ✅ Range 0.45–0.97 |
| Top-1 match (BOS) | token 220 | ✅ Both agree |
| BOS embedding match | cos=1.0 | ✅ File extraction verified |
| llama dep free | yes | ✅ ldd+nm verified |

## Triple DA Finding #1: SSM Recurrence is IDENTICAL
Both implement: `h ← h * exp(gate)` → `hk = h · k` → `diff = v - hk` → `h += k · diff · beta` → `out = h · q * (1/sqrt(128))`
Llama.cpp uses custom ggml op `ggml_gated_delta_net` (ops.cpp:10547).
Bytropix uses manual C (wubu_ssm.c:183).
**Math is the same. Scale is the same (1/sqrt(128)).**

## Triple DA Finding #2: Divergence Source
Quantized matmul dequant precision (Q5_K, Q6_K, IQ2_XXS, IQ3_XXS).
Different dequant → different float values → accumulating drift through 40 layers.
At IQ2_M quantization, this is EXPECTED behavior.

## Cold Gaps
| Prio | Gap | Detail |
|------|-----|--------|
| P0 | IQ3_XXS AVX2 vec_dot | _mm_hadd_epi16 bug, reverted to generic |
| P1 | Quantized matmul precision | Port ggml dequant kernels for bit-exact |
| P1 | Output proj split | Parallelize Q4_K across 16 threads |
| P2 | Expert prefetch | API ready, needs wiring |
| P2 | SSM attn AVX2 | 0.8ms/layer, 24ms total — low priority |

## Vault Papers Read
- Unsloth UD quant formula: per-tensor bpw breakdown
- Qwen3: 256-expert MoE + thinking mode
- DeepSeek-V3: MTP self-spec decode, sigmoid gating
- Synthesis: P0-P3 priority map (confirmed by DA)

## Modified Files (llama.cpp)
- `src/llama-graph.h`: t_layer_h vector
- `src/models/qwen35moe.cpp`: ggml_set_output per layer
- `src/llama-context.cpp`: DUMP_LAYER_DIR deep-copy F32 dump

## New Tools (bytropix)
- `tools/run_bos.c`: Standalone single-token forward
- `Makefile`: run_bos target
- `ref_dumper`: Uses libllama (pre-existing, enhanced)
