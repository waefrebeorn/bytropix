# WuBuText AI — Entry Point

## Purpose
Build commands, hardware spec, quick-start.

---

## Hardware
- **GPU:** NVIDIA RTX 5050, 6.4GB VRAM, sm=120
- **NVCC:** /usr/local/cuda-13.1/bin/nvcc -arch=sm_120
- **CPU:** AMD 16+ cores, 25GB RAM

## Build
```bash
make infer_moe_lazy      # Lazy MoE dequant (S1)
make infer_unified       # 40-layer SSM→GQA→MoE (S2)
make test_kv_cache       # GQA KV cache test (S3)
make infer_vision_gpu    # GPU vision 256x256 in 217ms
make infer_poincare      # GPU Poincare SSM 2835 tok/s
make test_moe            # MoE forward test
make train_real          # CPU training pipeline
```

## File Layout
```
src/          — Core: ssm, moe, model, gguf_reader, cuda_kernels, vision
include/      — Headers
tools/        — infer_moe_lazy, infer_unified, test_kv_cache, test_* 
data/          — embeddings, tokenizer, training data
.hermes/       — Mind palace, research, references
/models/       — GGUF (Qwen3.6-35B-A3B-UD-IQ2_M.gguf)
```
