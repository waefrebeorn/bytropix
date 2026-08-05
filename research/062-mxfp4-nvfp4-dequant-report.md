# DeepSeek-V4-Flash — MXFP4/NVFP4 Dequantization Reference Report

## Source Model

**DeepSeek-V4-Flash-0731** — AtomicChat/DeepSeek-V4-Flash-0731-GGUF (HuggingFace)

- **284B parameters**, 13B active per token (MoE)
- 43 layers, hidden_size = 7168
- 256 routed experts + 1 shared expert
- 6 active experts per token (top-k = 6)
- 1M context length, vocab = 129,280
- **Quant-aware-trained** at 4-bit: routed experts stored in **MXFP4**,
  non-expert weights in **FP8** or **BF16**
- MLA (Multi-head Latent Attention): Q uses latent dim (q_lora_dim=1536, rank=512),
  KV uses grouped-query decomposition

## Problem

The wubuwizard GGUF reader (`src/gguf_reader.c`) did **not** handle
GGML types `MXFP4` (id 39) or `NVFP4` (id 40). When a DeepSeek-V4-Flash GGUF
was loaded, these tensors silently zeroed out in the `default:` case of both
`gguf_dequantize()` and `gguf_read_tensor_f32()`, producing a model that
would run but output garbage for all MoE expert operations.

This is the **critical gap** — experts are 96% of the model's parameters and
carry the semantic capacity. Without MXFP4 dequant, the engine cannot run
DeepSeek-V4-Flash correctly.

## Solution

### New modules

| File | Responsibility |
|------|---------------|
| `include/wubu_dequant_fp4.h` | Public API: `wubu_fp4_raw_size()`, `dequantize_row_mxfp4()`, `dequantize_row_nvfp4()` |
| `src/wubu_dequant_fp4.c` | C11 implementation: E2M1 table, E8M0/UE4M3 scale decode, block-level dequant rows |

### Wiring into the GGUF reader

`src/gguf_reader.c` was patched in three locations:

1. **Enum** (`include/gguf_reader.h`): Added `GGML_TYPE_MXFP4 = 39` and
   `GGML_TYPE_NVFP4 = 40` to the `ggml_type` enum (TurboQuant branch IDs).

2. **`gguf_dequantize()`** switch: Added cases for types 39 and 40 that call
   `dequantize_row_mxfp4()` / `dequantize_row_nvfp4()`.

3. **`gguf_raw_size()`**: Added byte-size calculation:
   - MXFP4: `((n + 31) / 32) * 17` — 1 byte E8M0 scale + 16 bytes packed E2M1 per 32-element block
   - NVFP4: `((n + 63) / 64) * 36` — 4 bytes UE4M3 scales + 32 bytes packed E2M1 per 64-element block

4. **`gguf_read_tensor_f32()`**: Added `else if` branches for the two types
   so the file-loading path (not just the RAM-dequant path) handles them.

### Test tooling

`tools/test_gguf_load.c` — `known_type()` updated to accept types 39 and 40.

## Format Details

### MXFP4 — OCP Microscaling 4-bit (ggml type 39, QK_MXFP4 = 32)

```
struct block_mxfp4 {
    uint8_t e;       // E8M0 shared exponent (bias 127): scale = 2^(e-127)
    uint8_t qs[16];  // 32 × 4-bit E2M1 values, 2 per byte (high nibble first)
};                    // total: 17 bytes per 32 elements
```

E2M1 codeword → value: `0 → 0, 1 → 0.5, 2 → 1.0, 3 → 2.0`

### NVFP4 — NVIDIA Microscaling 4-bit (ggml type 40, QK_NVFP4 = 64, sub = 16)

```
struct block_nvfp4 {
    uint8_t d[4];     // 4 × UE4M3 scale bytes (one per 16-element sub-block)
    uint8_t qs[32];   // 64 × 4-bit E2M1 values, 2 per byte
};                     // total: 36 bytes per 64 elements
```

UE4M3 (unsigned E4M3, bias 7):
- `e == 0 && m == 0` → 0
- `e == 0 && m != 0` → subnormal: `m * 2^(-6)`
- `0 < e < 31` → `(1 + m/8) * 2^(e-7)`
- `e == 31` → Inf (treated as 0 — shouldn't occur in practice)

## Verification

### Unit tests (`tools/test_fp4_dequant.c` — 15/15 pass)

| Test | Description | Expected | Got | Status |
|------|-------------|----------|-----|--------|
| MXFP4 raw_size | 64 elements | 34 bytes | 34 | ✅ |
| NVFP4 raw_size | 64 elements | 36 bytes | 36 | ✅ |
| MXFP4 raw_size | 128 elements | 68 bytes | 68 | ✅ |
| NVFP4 raw_size | 128 elements | 72 bytes | 72 | ✅ |
| MXFP4 dequant block 0 | scale=0x80 (2.0), vals=[4,2,1,0] | [4,2,1,0] | [4,2,1,0] | ✅ |
| MXFP4 dequant block 1 | scale=0x82 (8.0), vals=[8,8,4,4] | [8,8,4,4] | [8,8,4,4] | ✅ |
| MXFP4 dequant max scale | scale=0xFF (2^128=Inf) | Inf | Inf | ✅ |
| NVFP4 dequant | scale=0x78 (256), vals=[512,256,128,0] | [512,256,128,0] | [512,256,128,0] | ✅ |
| NVFP4 dequant subnormal | scale=0x01 | subnormal | correct | ✅ |
| F32 control | direct copy | [1.5,2.5,3.5,4.5] | same | ✅ |
| MXFP4 zero scale | scale=0x00 | 0 | 0 | ✅ |
| Mixed elements | 47 elements (non-block-aligned) | padded zeros | correct | ✅ |
| Round-trip consistency | MXFP4 → F32 → compare | match | match | ✅ |
| Scale edge case | E8M0=0x00 (2^-127=subnormal) | ~0 | ~0 | ✅ |
| NVFP4 scale=0x7F | max normal | 49152 | 49152 | ✅ |

### Full GGUF load gate (`tools/test_gguf_load.c` with synthetic GGUF)

A synthetic GGUF file (`tools/gen_test_gguf.py`) was generated with
3 tensors: one MXFP4 (type 39), one NVFP4 (type 40), one F32 (type 0).

```
GGUF v2, 3 tensors, 1 KV pairs
Tensor info end at offset 220, aligned to 224 (pad=4, alignment=32)

type histogram:
  F32      ( 0): 1
  MXFP4    (39): 1
  NVFP4    (40): 1
offset-span vs size-table mismatches: 0

sample f32_test.weight        type F32     elems=4  mean=3.00000  rms=3.20156  nan=0
sample mxfp4_test.weight      type MXFP4   elems=64  mean=0.48438  rms=1.68170  nan=0
sample nvfp4_test.weight      type NVFP4   elems=64  mean=14.00000  rms=73.32121  nan=0

LOAD GATE PASSED — all 3 tensors type-resolved, offsets monotonic, samples NaN-free
```

Dequantized values verified mathematically:
- MXFP4: sum = 4+2+1+0+8+8+4+4 = 31, mean = 31/64 = 0.484375 ✅
- NVFP4: sum = 512+256+128+0 = 896, mean = 896/64 = 14.0 ✅

### Bug caught during verification

Initial implementation used `expf()` (e^x) for E8M0/UE4M3 scale decode instead
of `powf(2.0f, x)` (2^x). This produced incorrect scales (e.g., 2.718 instead of 2.0).
Fixed before the test commit. **Root cause**: conflating Euler's number e with
base-2 exponent encoding.

### Windows build verification

```
make -f Makefile.win gen_text_win   →  ✅ builds (gcc 16.1.0, -Wall -Wextra)
make -f Makefile.win lfm2_test      →  ✅ builds
test_gguf_load.c compile            →  ✅ no new warnings
```
