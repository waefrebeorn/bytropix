# `llama/` — External References & Resources

Reference implementations and research resources for inference engineering.

## Directories

| Directory | Source | Purpose |
|-----------|--------|---------|
| `ggml-common.h` | llama.cpp | Shared GGML type definitions |
| `ggml-cpu/` | llama.cpp | CPU backend reference |
| `ggml-cuda/` | llama.cpp | CUDA backend reference |
| `llama-model.cpp` | llama.cpp | Model architecture reference |
| `src/` | llama.cpp | Reference source files |
| **`turboquant_plus/`** | [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) | Google TurboQuant KV cache compression — PolarQuant + WHT |
| **`rotorquant/`** | [scrya-com/rotorquant](https://github.com/scrya-com/rotorquant) | RotorQuant — block-diagonal Clifford rotor KV cache compression |

## KV Cache Compression Resources

Three approaches for fitting 256k+ context on laptop GPUs:

### 1. Q4_0 (Phase 22) — Implemented ✅
- **File:** `include/wubu_model.h` — `KV_CACHE_Q4_0` mode
- **Compression:** 4:1 vs F16 (720 MB vs 2.56 GB at 256k)
- **Status:** CPU path working (cos-sim 0.9994). GPU path pending.

### 2. TurboQuant+ (`turboquant_plus/`)
- **Paper:** Google ICLR 2026
- **Method:** Walsh-Hadamard Transform + PolarQuant
- **Claimed:** 3.8-6.4× compression, near-q8_0 PPL, zero speed penalty
- **Relevance:** WHT spreads outlier energy before quantization → better quality at same bitrate

### 3. RotorQuant (`rotorquant/`)
- **Paper:** Scrya Research
- **Method:** Block-diagonal Givens rotation (2×2) or quaternion rotation (4×4)
- **Claimed:** Beats TurboQuant on every axis: better PPL, 28% faster decode, 44× fewer params
- **Relevance:** Simpler than WHT — 2 FMAs per pair. Easier to integrate.

### 4. Hamilton Encoder Attention (Legacy — HASHMIND Project)
- **Location:** `/mnt/c/projects/HASHMIND/llama-cpp-rotorquant/llama.cppCOPY/`
- **Method:** MLP encoder compresses V cache to 5D quaternion grid, BSP tree for O(log N) retrieval
- **Status:** Experimental, 4096 recall window bug documented
- **Relevance:** Orthogonal to quantization — compresses entire representations, not individual values

## Examples

| File | Description |
|------|-------------|
| `tools/example_rotorquant.py` | RotorQuant Givens rotation + Q4_0 quantization demo |
| `tools/example_turboquant.py` | TurboQuant WHT + Q4_0 quantization demo |
| `tools/example_hamilton_encoder.py` | Hamilton encoder quaternion manifold compression demo |

## Full Resource Doc

See `vault/cache-compression-resources.md` for detailed comparison, integration paths, and block formats.
