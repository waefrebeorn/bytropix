# WuBu GGUF Reader — Phase 1 Implementation ✅ DONE

## Goal
Extract `token_embd.weight` and `output.weight` tensors from Qwen3.6-35B-A3B GGUF file.
Then analyze embedding distribution and map to Poincaré ball.

## Status
All steps complete. See `README.md` for details.

## Key Files That Exist
| File | Purpose | Status |
|------|---------|--------|
| `src/gguf_reader.c` | GGUF parsing + Q5_K dequant + Poincaré ops | ✅ |
| `include/gguf_reader.h` | Public API | ✅ |
| `tools/extract_and_map.c` | CLI tool | ✅ |
| `tools/extract_embeddings.py` | Python verifier | ✅ |
| `tools/analyze_embeddings.py` | NN preservation test | ✅ |
| `data/qwen36_embeddings_c.bin` | Mapped embeddings | ✅ (2.03GB) |
| `data/qwen36_embeddings_c.bin.raw` | Raw dequantized | ✅ (2.03GB) |

## GGUF Format (from llama.cpp gguf.h)

[4B] magic = "GGUF"
[4B] version (uint32, currently 3)
[8B] tensor_count (int64)
[8B] kv_count (int64)
... KV pairs ...
... tensors ...
... data blob ...

Each tensor entry:
[8B] name_length (uint64) + name bytes
[4B] n_dims (uint32)
[n_dims × 8B] dims (int64 each)
[4B] tensor_type (ggml_type enum as int32)
[8B] data_offset (uint64, relative to start of data blob)

## Tensor Names Found in Qwen3.6-35B-A3B GGUF

token_embd.weight  → [2048, 248320] stored, transpose to [248320, 2048] — Q5_K (type 13)
output.weight      → [2048, 248320] — Q6_K (type 14)
blk.N.attn_qkv.weight  → [2048, 8192] — Q8_K (fused Q+K+V)
blk.N.attn_gate.weight → [2048, 4096] — Q8_K (output gate)
blk.N.ssm_conv1d.weight → [4, 8192] — F32 (DeltaNet conv kernel)
blk.N.ssm_alpha.weight → [2048, 32] — F32 (DeltaNet alpha)
blk.N.ssm_beta.weight → [2048, 32] — F32 (DeltaNet beta)
blk.N.ffn_gate_exps.weight → [2048, 512, 256] — IQ2_XS (MoE gate proj)
blk.N.ffn_up_exps.weight  → [2048, 512, 256] — IQ2_XS (MoE up proj)
blk.N.ffn_down_exps.weight → [512, 2048, 256] — IQ1_S (MoE down proj)
blk.N.ffn_gate_inp.weight → [2048, 256] — F32 (router)

## Quantization Types Discovered
| Type ID | Name | Used For |
|---------|------|----------|
| 0 | F32 | Norms, biases, router, SSM params |
| 12 | Q4_K | output.weight |
| 13 | Q5_K | token_embd.weight |
| 14 | Q6_K | Some MoE shared expert weights |
| 15 | Q8_K | Attention weights, shared expert FFN |
| 17(=actually13) | Q5_K (mis-ID'd) | token_embd.weight — NOT IQ2 |
| 18 | IQ1_S | MoE down projections |
| 19(=17) | IQ2_XS | MoE gate/up projections |

Note: Tensor quantization varies per-tensor within a single GGUF file. The "IQ2_M" file
only means experts are IQ2 — embeddings are kept at Q5_K (higher precision).

