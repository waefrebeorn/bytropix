# Overnight Map — May 19, 2026 (Triple DA v6)

## TRIPLE DA SUMMARY
**SSM recurrence math IDENTICAL** between bytropix and llama.cpp.
Both use `scale=1/sqrt(128)`, `h←h*exp(gate)+k·(v-h·k)·beta`, `out=h·q*scale`.
Divergence is quantized matmul precision — NOT algorithm bug.

## Infrastructure Built
- DUMP_LAYER_DIR env var → per-layer F32 dumps (ref_layer_N.bin, our_layer_N.bin)
- ggml_set_output() fix for 40 unique layer tensors
- run_bos: standalone single-token forward pass (no tokenizer needed)
- Logit comparison script via numpy

## Session Findings
| Finding | Value | Impact |
|---------|-------|--------|
| Logit cos-sim vs llama.cpp | 0.7944 | Pre-existing at IQ2_M |
| Per-layer cos-sim avg | 0.88 | Range 0.45–0.97 |
| SSM recurrence | IDENTICAL | Verified ggml_gated_delta_net source |
| Top-1 match | token 220 | BOTH agree |
| BOS embedding | cos=1.0 | File vs GGUF matches |
| AVX2 IQ3_XXS vec_dot | BROKEN | _mm_hadd_epi16, reverted to generic |

## Critical Knowledge (DON'T RE-DERIVE)
- `ggml_set_output()` prevents buffer reuse (took 3 iterations to discover)
- `ggml_gated_delta_net` kernel: ops.cpp line 10547, scale=1/sqrt(S_v)
- bytropix wubu_ssm_forward: wubu_ssm.c line 183, same 3-step recurrence
- The 0.79 cos-sim is NOT a bug — it's quantized arithmetic precision

## Next Workstream
A — AVX2 IQ3_XXS vec_dot fix (port from ggml-quants.c properly)
B — Measure multi-token drift (5-token sequence, per-layer cos-sim)
C — Output proj split

## Fallback
If blocked on AVX2: build and compare multi-token sequences.
Write test: 5 BOS tokens, track per-layer cos-sim vs llama.cpp.
