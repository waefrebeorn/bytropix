# `include/` — C Headers

**Headers for the inference engine. KV cache, layer weights, model structs.**

| File | Lines | Purpose |
|------|-------|---------|
| `wubu_model.h` | ~300 | Model struct, KV cache helpers (Q4_0/F16/F32), MTP head struct |
| `wubu_ssm.h` | ~371 | SSM/GQA weight structs, forward declarations |
| `wubu_moe.h` | ~117 | MoE constants (256 experts, 8 active), weight struct |
| `gguf_reader.h` | 144 | GGML type enums, reader API, block_q8_K definition |
| `wubu_mobius.h` | — | Möbius/Poincaré math functions (experimental) |
| `gpu_output_proj.h` | ~30 | GPU output projection declarations |
| `cuda_kernels.h` | — | GPU kernel declarations (attention, SSM, parallel scan) |
| `gpu_ssm_recurrence.h` | — | GPU SSM recurrence kernel declarations |
| `gpu_moe_kernel.h` | — | GPU MoE kernel declarations |

## Key: Q4_0 KV Cache (`wubu_model.h`)

The KV cache supports three modes controlled by preprocessor defines:

```c
#define KV_CACHE_Q4_0  // 4-bit quantized (default): 0.56 bytes/elem
// #define KV_CACHE_F16  // Half-precision: 2 bytes/elem
// (neither)           // Full float: 4 bytes/elem
```

Block format: `block_q4_0_cache {uint16_t d, uint8_t qs[16]}` — 32 elements per block.
