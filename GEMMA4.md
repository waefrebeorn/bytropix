# bytropix — Gemma 4 12B Engine

## Architecture
- Dual-head-dim ISWA: 40 sliding (HDIM=256, Q=4096, KV=2048, 8KV heads, 10K rope) + 8 full (HDIM=512, Q=8192, KV=512, 1KV head, 1M rope, 25% rotary, rope_freqs)
- Full attn at indices: 5,11,17,23,29,35,41,47 (every 6th from 5)
- KV sharing: layers 40-47 reuse from 38-39

## Files

| File | Purpose |
|------|---------|
| `include/wubu_gemma4.h` | Model structs, per-layer dimension helpers, API |
| `include/gpu_gemma4.h` | GPU context, kernel API |
| `include/gguf_reader.h` | GGUF parsing, Q4_K/Q4_0 dequant, quantized_matmul API |
| `src/wubu_gemma4_model.c` | CPU forward: g4_batched_qmatmul (AVX2 Q4_K vec_dot), g4_layer_forward_cpu, g4_model_forward |
| `src/gpu_gemma4.cu` | GPU kernels: sliding_attn, full_attn, rms_norm, rope, gelu, element-wise |
| `src/gpu_gemma4_forward.cu` | GPU forward orchestrator: weight upload, Q4_K dequant kernel, full forward loop |
| `src/quantized_matmul.c` | CPU Q4_K matmul: Q8_K quant + SIMD vec_dot (OpenMP) |
| `src/quantized_dot_generic.c` | SIMD vec_dot for Q4_K (SSSE3/AVX2) |
| `src/gguf_reader.c` | GGUF I/O, all quant type dequant functions |
| `tools/test_gemma4.c` | CPU test (make test_gemma4) and GPU test (make test_gemma4_gpu) |

## Build
```bash
make test_gemma4          # CPU binary
make test_gemma4_gpu      # GPU binary (links cuda)
```

## GPU Run
```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-13.1/targets/x86_64-linux/lib \
OMP_NUM_THREADS=8 taskset -c 0-7 \
./test_gemma4_gpu /home/wubu/models/gemma4/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf
```

## Perf
- **CPU**: 0.6 tok/s (HW-limited: i7-14650HX, 7×6.4GB reads at DDR5)
- **GPU**: Same (GPU weight upload works, compute still CPU — GPU kernels WIP)

## GPU Compute — Next Steps
llama.cpp achieves 42 tok/s on RTX 5050 via `mmq.cuh` — a 4176-line templated CUDA engine that:
1. Keeps Q4_K weights on GPU at init (6.4GB fits in 8GB)
2. Fuses dequant + matmul into warp-level tile-based kernels (no F32 intermediate)
3. Never touches CPU for layer compute

Our GPU path needs:
- [ ] Wire Q4_K dequant kernel into actual forward (currently just uploaded)
- [ ] Replace CPU g4_batched_qmatmul calls with GPU dequant+cuBLAS
- [ ] Wire GPU RMSNorm + RoPE + attention kernels into layer loop
- [ ] Full GPU forward: all data stays on GPU, LM head on GPU

## MTP Draft Model
- `mtp-gemma-4-12B-it.gguf`: hidden=1024, FFN=8192, 2-layer draft
- Not started yet. Needs: model loader, forward pass, speculative decoding loop

## Models
- `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` — Main 12B (6.4GB)
- `mtp-gemma-4-12B-it.gguf` — MTP draft (~0.3GB)
- `mmproj-F16.gguf` — Vision encoder (11 tensors)
