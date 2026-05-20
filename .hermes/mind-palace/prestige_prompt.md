# Prestige Prompt — May 19, 2026 PM (Phase 22 — Q4_0 KV Cache ✅)

## Project: bytropix — Qwen3.6-35B-A3B-UD-IQ2_M
**Overall cos-sim: 0.9994 vs llama.cpp (CPU, 5-token, 40 layers).**
**CPU prefill: ~12 tok/s. GPU: ⚠️ gen_text_gpu hangs. Q4_0 KV cache: 4:1 compression.**

## ARCHITECTURE CORRECTION (May 19)
The true architecture is a **3:1 SSM/GQA interleaved repeating pattern**, NOT the previously assumed "30 SSM + 10 GQA contiguous". Confirmed via:
- GGUF tensor `blk.N.ssm_a` vs `blk.N.attn_q.weight` presence
- llama.cpp `full_attention_interval=4` metadata key
- DUMP_INTERMEDIATE_DIR per-layer tensor naming

SSM layers: 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38
GQA layers: 3,7,11,15,19,23,27,31,35,39

## Phase 22: Q4_0 KV Cache Compression ✅
### What
New `KV_CACHE_Q4_0` mode in `wubu_model.h`. Stores K/V cache in 4-bit quantized blocks:
- `block_q4_0_cache {uint16_t d, uint8_t qs[16]}` — 32 elements per block, 18 bytes
- 4:1 compression vs F16: 720MB vs 2.56GB at 256k
- Fixed `kv_cache_read_head` for arbitrary-length multi-block reads
- `kv_cache_write_head` with aligned bulk write path

### Cos-sim: 0.9994 overall (identical to F16)
- L00-L30: 0.998-0.9999
- L31 (GQA-only): 0.9585 — quantization noise amplification through 30 layers
- L32+: recovers

### Verified
- CPU `gen_text` + Q4_0: same output as F16
- Prefill test: 5 tokens, 40 layers, cos-sim 0.9994
- GPU cache stays FP16 (native cuBLAS format)

## DUMP_INTERMEDIATE_DIR: Per-Operation Reference Tracing ✅
Modified llama.cpp's `llm_graph_context::cb()` to save ALL intermediate tensors:
- 53 unique tensor names per layer: `conv_input`, `conv_output_silu`, `linear_attn_out`, `Qcur`, `Kcur`, `Vcur`, `beta`, `alpha`, `gate`, `new_state`, `state_predelta`, `attn_output`, `ffn_moe_*`, `l_out`, etc.
- 1997 F32 files per 5-token forward pass
- Environment variable: `DUMP_INTERMEDIATE_DIR=/path/to/dir`

## Bug #13: kv_cache_read_head Multi-block Read Fixed
- Root cause: Q4_0 read path assumed max 2 blocks (64 elements)
- Full head read (256 elements) spans 8 blocks — caused hang
- Fix: while-loop with per-block dequant + partial copy

## Remaining
- P0: gen_text_gpu hang (pre-existing, unrelated to Phase 22)
- P0: GPU Q4_0 KV cache (FP16 → Q4_0, saves 3.7GB VRAM)
- P1: Unified SSM kernel fusion
- P2: Sparse attention with global tokens for 512k+
