# WuBuText AI — Entry Point (May 21 PM v2)

## TRUTH
**CPU inference WORKS.** Sequential SSM (FORCE_CPU_SSM_SEQ=1) produces coherent text.
**GPU inference builds** but is net-negative (slower than CPU).
**GPU vision encoder: 0.52s ViT** (122x vs CPU) — the only GPU win.

## Hardware
- **GPU:** NVIDIA RTX 5050 (Blackwell sm_120), 6.5-8 GB VRAM
- **NVCC:** /usr/local/cuda-13.1/bin/nvcc -arch=sm_120
- **CPU:** AMD 12+ cores, 16 GB RAM
- **CUDA toolkit:** 13.1

## Build & Run
```bash
# CPU inference (USE THIS — works correctly)
make gen_text_cpu
FORCE_CPU_SSM_SEQ=1 ./gen_text_cpu "The capital of France is" 20 40

# GPU inference (builds, slower than CPU)
make gen_text_gpu
GPU=1 FORCE_CPU_MOE=1 ./gen_text_gpu "prompt" 20

# MTP speculative decode (needs rebuild)
make gen_text_mtp
MTP=1 ./gen_text_mtp "prompt" 20

# Compare with reference
tools/ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "prompt" 0
tools/layer_cos_sim /tmp/ref /tmp/our 40

# Vision test
make test_vision_real
./test_vision_real <mmproj.gguf> <pixels.bin>
```

## Key Env Vars
```
FORCE_CPU_SSM_SEQ=1     # Force sequential SSM (coherent output)
ROPE_SCALE_FACTOR=0.25  # 4x context extension
USE_SPARSE_ATTN=1        # NSA sparse attention
SPARSE_W=512 SPARSE_G=128 # Sparse attention params
MTP=1                    # Enable MTP speculative decode
GPU=1 FORCE_CPU_MOE=1    # Enable GPU hybrid (SSM/GQA on GPU, MoE on CPU)
```

## File Layout
```
src/          — SSM, MoE, model, gguf_reader, CUDA kernels, vision
include/      — Headers
tools/        — Inference binaries, test harnesses, utilities
.hermes/       — Mind palace, vault, research
/models/       — Qwen3.6-35B-A3B-UD-IQ2_M.gguf (only model)
~/llama.cpp/   — Reference implementation (libllama.so + llama-cli)
```

## Related Repos
- ~/llama.cpp/ — Reference GGUF inference (modify for inline hooks)
- ~/HASHMIND/tailslayer/ — Hedged-read C++ (spec-decode inspiration)
- ~/HASHMIND/llama-cpp-rotorquant/ — Hamilton encoder CUDA kernels

## Known Blockers
1. **Chunked SSM CS>1**: Must use FORCE_CPU_SSM_SEQ=1
2. **GPU net-negative**: H2D/D2H overhead
3. **L31 cos-sim 0.9585**: GQA attention divergence
