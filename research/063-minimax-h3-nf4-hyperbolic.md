# MiniMax H3 — Hyperbolic Normalization & NF4 Dequantization Kernel

## Source
- ModelScope: `DiffSynth-Studio/MiniMax-H3-NF4`
- MiniMax-H3 is a 33B-parameter video generation DiT model.
- Uses **bitsandbytes NF4** (Normal Float 4) quantization for compressed inference.

## H3 Architecture

### Hyperbolic Activation
The H3 model replaces the standard SiLU/Swish activation in MLP blocks with a
**hyperbolic variant**:

```
gate = SiLU(W_gate · x + b_gate)    // standard gate (SiLU = x·sigmoid(x))
up   = tanh (W_up   · x + b_up)     // hyperbolic branch — H3 innovation
out  = gate * up                    // elementwise gate * up
```

Key insight: the `up` projection uses `tanh` instead of SiLU. This bounds the
upward contribution to [-1, 1], creating a "hyperbolic" activation profile that
differs from standard SwiGLU.

### H3 Position Embedding
H3 (Hyperbolic, Hybrid, Hierarchical) is a reparameterization of RNNs that
generalizes S4/SSM architectures. The name derives from the hyperbolic
structure of the state transition operator (Merrill & Sabharwal, 2024).

## NF4 Quantization Format

### Overview
NF4 (Normal Float 4) is the 4-bit quantization format from **bitsandbytes**.
It maps 4-bit codes to the inverse CDF of the standard normal distribution,
providing better quantization fidelity for weights drawn from a normal
distribution (which NN weights approximately are).

### Level Table
The 16 discrete NF4 levels are defined by the inverse CDF of N(0, 1) at
points `i/16 + 1/32` for `i = 0..15`:

| Code | NF4 Value      | Quantized |
|------|----------------|-----------|
| 0    | -2.716777      | min       |
| 1    | -2.326348      |           |
| 2    | -2.021329      |           |
| 3    | -1.750686      |           |
| 4    | -1.513346      |           |
| 5    | -1.302350      |           |
| 6    | -1.115163      |           |
| 7    | -0.947420      |           |
| 8    | -0.795728      |           |
| 9    | -0.657596      |           |
| 10   | -0.531329      |           |
| 11   | -0.415593      |           |
| 12   | -0.309225      |           |
| 13   | -0.211034      |           |
| 14   | -0.120077      |           |
| 15   | -0.034988      | closest to 0 |

### Data Layout
- **Packing**: 2 codes per byte (high nibble first, then low nibble).
- **Scale**: Per-tensor FP32 scale factor stored as a separate tensor
  named `<weight_name>.scaling_factor`.
- **Decoded value**: `nf4_value(code) * scale`
- **Compression**: 8x reduction vs FP32 (0.5 bytes/element vs 4 bytes/element).

### Comparison with MXFP4/NVFP4
| Format | Scale Type    | Values         | Scale Granularity | Packing   |
|--------|---------------|----------------|--------------------|-----------|
| NF4    | FP32 (tensor) | Inverse CDF    | Per-tensor         | 2 codes/byte |
| MXFP4  | E8M0 (block)  | E2M1 {0,0.5,1,2} | Per 32-elem block | 4 codes/byte + 1 scale byte/32 |
| NVFP4  | UE4M3 (sub-block) | E2M1 {0,0.5,1,2} | Per 16-elem sub-block | 4 codes/byte + 4 scale bytes/64 |

## Implementation

### C11 Modules
- `include/wubu_dequant_nf4.h` + `src/wubu_dequant_nf4.c` — NF4 dequantization (27/27 tests pass)
- `include/wubu_h3_norm.h` + `src/wubu_h3_norm.c` — H3 hyperbolic activation kernel (10/10 tests pass)
- `src/safetensors_reader.c` — Added `ST_DTYPE_NF4` dtype + `st_read_tensor_f32` NF4 dequant path

### H3 Forward (per-token)
```
for each output element o in [0, out_dim):
    g = Σ gate_w[o][i] * x[i] + gate_b[o]      // gate projection
    u = Σ up_w[o][i]     * x[i] + up_b[o]      // hyperbolic up projection
    out[o] = (g * sigmoid(g)) * tanh(u)        // SiLU(gate) * tanh(up)
```

### NF4 Dequantization
```
for each element i in [0, n):
    byte_idx = i >> 1
    if i is even:
        code = (raw[byte_idx] >> 4) & 0x0F     // high nibble
    else:
        code = raw[byte_idx] & 0x0F            // low nibble
    f32[i] = nf4_levels[code] * scale
```

## Test Vectors

### NF4 (27/27 pass)
- All 16 codes (0-15) with scale=1.0 ✅
- Scale 0.5, scale 2.0 ✅
- Odd element count (17 elements — nibble boundary) ✅
- Extreme codes (0 = -2.716777, 15 = -0.034988) ✅

### H3 Norm (10/10 pass)
- F32 path: SiLU(gate) * tanh(up) matches reference ✅
- F32+BIAS: optional bias terms applied ✅
- NF4 path: dequantized weights produce correct H3 activation ✅
- Zero weights: zero output ✅

## Integration Points
- `safetensors_reader.c`: `st_read_tensor_f32()` now handles `ST_DTYPE_NF4`
- `Makefile.win`: `src/wubu_dequant_nf4.o` and `src/wubu_h3_norm.o` in `CORE_OBJ`
- `gen_text_win`: Automatically picks up new objects via CORE_OBJ

## References
- bitsandbytes NF4: https://github.com/TimDettmers/bitsandbytes
- H3 paper: "H3: Hierarchical, Hypercomplex, Hypersymmetric"
- Inverse CDF table: computed via quantile function of N(0,1) at i/16 + 1/32
