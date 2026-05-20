# Overnight Map — Phase 25

**Active repo**: /home/wubu/bytropix/
**Model**: Qwen3.6-35B-A3B-UD-IQ2_M.gguf (qwen35moe arch)
**Binary**: ./gen_text_gpu (GPU=1, MAX_CTX=262144)
**Ref binary**: /home/wubu/llama.cpp/build/bin/llama-cli
**Current rate**: ~8.5 tok/s decode (4K ctx), 4.8 tok/s (256k)
**VRAM**: ~3.56GB total, fits 6.5GB GPU

## Modified files
- src/gpu_quant_matmul.cu — fused Q5_K + Q6_K matmul (no bv[256] spill)

## Vault
- vault/tmp-tools/phase25/ — current source copies
- vault/deepseek-collection/ — 28 PDFs (V3, V3.2, NSA, MoE, V4, etc.)
- vault/qwen36-repo/ — Qwen3.6 README

## Next step
- Phase 26: Fuse SSM post-matmul ops for N=1 decode (cuBLAS beta/alpha → manual dot, fuse element-wise ops)

## Build
$ make gen_text_gpu  # Clean build, -arch=sm_120
$ GPU=1 MAX_CTX=4096 GPU_QUANTIZED=1 ./gen_text_gpu "prompt" N

## Env vars
- GPU=1 — enable GPU inference
- MAX_CTX=N — max context size
- GPU_Q4_0_KV=0 — use FP16 KV cache (default Q4_0)
- GPU_QUANTIZED=1 — Q4_K output proj mode
- GQA_WINDOW=N — sliding window attention
- DUMP_INTERMEDIATE_DIR=/tmp/dump — llama.cpp ref tensor dump
