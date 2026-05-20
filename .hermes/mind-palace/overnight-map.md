# Overnight Map — Phase 22: Q4_0 KV Cache + Architecture Discovery

## State: Overall cos-sim 0.9994 (CPU, 5-token). Q4_0 KV cache 4:1. GPU hang pre-existing.

Phase 22 complete: **Q4_0 KV cache compression (4:1 vs F16)** and **architecture discovery (3:1 interleaved SSM/GQA)**.

## Quick Trunk Reference
- Source: `/home/wubu/bytropix/`
- Model: `/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf`
- Build: `make gen_text` (CPU) or `make gen_text_gpu` (GPU)
- Reference: `DUMP_LAYER_DIR=/tmp/ref_layers ./ref_dumper model.gguf "prompt" 0`
- Compare: `tools/layer_cos_sim /tmp/ref_layers /tmp/our_layers 40`
- Intermediates: `DUMP_INTERMEDIATE_DIR=/tmp/ref_int ./ref_dumper model.gguf "prompt" 0`

## What Was Done
- **DUMP_INTERMEDIATE_DIR**: llama.cpp modified to dump 53 tensor types/layer (1997 files)
- **Architecture discovery**: GGUF tensor enumeration proved 3:1 interleaved pattern
- **Phase 22: Q4_0 KV cache**: block_q4_0_cache, 4:1 compression, kv_cache_read/write_head fixed
- **Bug fix**: kv_cache_read_head now handles arbitrary-length reads (not just 2 blocks)
- **ref_dumper enhancement**: multi-token prompt support, numeric token ID mode
- **DA audit**: 3 stale docs fixed, all vault claims verified, propagation sweep done

## Workstreams (Pick One)
A — [P0] **Fix gen_text_gpu hang**: Debug pre-existing GPU inference hang. Check GPU init sequence, SSM full forward, or tokenizer fallback.
B — [P0] **GPU Q4_0 KV cache**: Port Q4_0 quantization to GPU growable cache. Currently FP16 (5.12GB). Saves ~3.7GB VRAM.
C — [P1] **Unified SSM kernel Phase A**: Fuse conv1d→SiLU→split→norm→beta into 1 kernel.

## Data Not To Re-Derive
- Architecture: 3:1 interleaved (SSM on layers 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38; GQA on 3,7,11,15,19,23,27,31,35,39)
- Q4_0 KV cache format: block_q4_0_cache {uint16_t d, uint8_t qs[16]} with aligned write path
- Cos-sim: L00-L30=0.998-0.9999, L31=0.9585 (GQA quantization noise), overall=0.9994
- VRAM with Q4_0: ~6,453 MB at 256k context
- GPU gen_text_gpu has pre-existing hang (was working before May 19 PM)

## Fallback
If stuck on GPU hang, do B (GPU Q4_0 KV cache — CPU-only testable) or work on docs/analysis.
