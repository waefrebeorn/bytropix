# Prestige Prompt — May 21, 2026 (Phase 28l: P1 Complete, P2 Up)

## Project: bytropix — Multi-Modal Inference (Text + Vision)
P1 complete: MTP spec decode working, vision→text pipeline verified.
P2: Feature cream (GPU RMSNorm, chunked prefill, sparse attn, RoPE, vision GPU).

## Benchmarks (Phase 28l Verified)
| Metric | Value | Status |
|--------|-------|--------|
| Decode (hybrid GPU SSM/GQA + CPU MoE) | 5.5 tok/s | ✅ Coherent text |
| MTP spec decode | 8.5 tok/s, 4% acceptance | ✅ |
| Vision→text pipeline | 256×256→128 patches→logits, no NaN | ✅ Verified |
| Vision encoder | 63.7s CPU (27 ViT layers) | ✅ Verified, needs GPU |
| GPU MoE v5 (single layer cos-sim) | 0.9888 | 🟡 Fundamental path diff |
| GPU MoE (40 layers) | 0.9968 → garbage | ❌ Accept hybrid |

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
