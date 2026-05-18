# bytropix — Quick Start (May 18 — Phase 2 Complete)

## Hardware
AMD Ryzen 7950X (16C/32T), 64GB DDR5, RTX 5050 6.4GB

## Model
/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (2.7 bpw, 10.7 GB)

## Build & Run
```bash
# Text generation
cd ~/bytropix && make gen_text
./gen_text "The capital of France is" 32

# Cos-sim verification vs llama.cpp
make test_full_moe
./test_full_moe

# Per-layer timing
PROFILE=1 ./test_full_moe

# Reference dumps (requires libllama.so)
make ref_dumper
LLAMA_DUMP_LAYERS=1 DUMP_LAYER_DIR=/tmp/ref ./ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf 248044
```

## Current Status
- Cos-sim: 0.9968 vs llama.cpp (T=1, single token)
- Decode: 0.6 tok/s (16 threads, CPU)
- Prefill: 1.0-1.4 tok/s
- Output: coherent English text
- All 40 layers > 0.995 cos-sim

## Key Config
- Compiler: gcc/g++ with -O3 -march=native -fopenmp
- OpenMP threads: 16 (auto-detected)
- Embedding cache: data/qwen36_embeddings_c.bin.raw (1.9 GB)
