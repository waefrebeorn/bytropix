# State — May 21, 2026 (Phase 28l: P1 Complete, P2 Up)

## COMPLETED P1
- ✅ MTP spec decode: gen_text_mtp at 8.5 tok/s (4% acceptance from quantized head)
- ✅ Vision pipeline verified: screenshot→encoder→mmproj→text→logits, no NaN
- ✅ 2 segfault bugs fixed in wubu_vision.c (n_patches_total cap, scores heap alloc)
- ✅ Makefile test_vision_real target fixed (GPU_SUPPORT linkage)
- ✅ test_vision_real builds and runs from `make test_vision_real`

## Current Reality
| Metric | Value | Status | Evidence |
|--------|-------|--------|----------|
| Hybrid decode (GPU SSM/GQA + CPU MoE) | 5.5 tok/s | ✅ Coherent text | gen_text_gpu GPU=1 FORCE_CPU_MOE=1 |
| MTP spec decode | 8.5 tok/s | ✅ Verified | gen_text_mtp "prompt" 30 |
| Vision→text pipeline | 256×256→logits, no NaN | ✅ Verified | test_vision_real with real screenshot |
| Vision encoder | 63.7s CPU, 27 ViT layers | ✅ Verified | Real 256×256 screenshot processed |
| GPU MoE (single layer) | 0.9888 cos-sim vs CPU | 🟡 Fundamental path diff | DA v13 analysis |

## Infrastructure Built This Session
1. `/home/wubu/bytropix/src/wubu_vision.c` — 2 segfault fixes (n_patches_total cap, scores heap)
2. `/home/wubu/bytropix/Makefile` — test_vision_real target fixed with GPU_SUPPORT
3. `/tmp/screen_vision_input.bin` — Test pixel data pipeline (ffmpeg→PIL→raw float)
4. `src/wubu_moe_cpu.o` — CPU-only moe object for GPU-free linking

## Cold Gaps (P2)
- GPU RMSNorm + SiLU + gated norm kernels
- Chunked prefill (3-7x speedup)
- NSA sparse attention
- RoPE extrapolation 4x
- GPU vision encoder kernels

## Critical Knowledge
- `ggml_set_output()` REQUIRED to prevent scheduler buffer reuse
- `ggml_gated_delta_net` is a custom GGML op — NOT manual C code in llama.cpp
- SSM recurrence uses `scale = 1/sqrt(S_v)` where S_v = 128 = ssm_d_state
- Quantized matmul dequant precision is the divergence source, not algorithm
- BOS 248044 for Qwen3.6-35B. Top-1 = 220 for single BOS forward.

## Tools Vault (tmp code copied)
- `run_bos` at `/home/wubu/bytropix/run_bos`
- `ref_dumper` at `/home/wubu/bytropix/ref_dumper`  
- Layer dumps: `/tmp/dump_layers_ref/` (llama.cpp), `/tmp/dump_layers_our/` (bytropix)
