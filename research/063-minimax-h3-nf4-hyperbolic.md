# MiniMax H3 — Hyperbolic Normalization, NF4 Dequant, & 3D MM-RoPE Kernel

## Sources
- HuggingFace: `MiniMaxAI/MiniMax-H3` (33B omni-modal: text, image, video, audio)
- ModelScope: `DiffSynth-Studio/MiniMax-H3-NF4` (NF4 quantized version)
- Twitter: `@tono_ken3/status/2084534748534748534415607` (technical commentary, JP)

## H3 Architecture (Corrected)

### Overview
MiniMax H3 is a **33B-parameter dense, single-stream omni-modal transformer**:
- **H3-Context-IR**: Multimodal instruction interpretation → Context IR
- **H3-Base**: 33B Omni-Transformer generating video+audio latents at 768p
- **H3-Regenerate-2K**: In-context 2K upscaling using H3-Base itself

### H3-Encoder
- Uses full pretrained **Qwen3-VL-32B** weights (50th layer hidden states)
- Adds special tokens like `<d>`

### H3-VisualVAE
- Temporal-causal video VAE: 16× spatial, 4× temporal compression, 24 latent channels
- Latent patches: `1×2×2` (time, height, width) → effective 32× spatial, 4× temporal

### H3-AudioVAE
- Stereo audio processing (independent channels, recombined)
- 32 kHz audio compressed to 40 Hz latent tokens

### H3-Omni-Transformer (33B)
- Dense, single-stream Transformer (~13B params in AdaLN branches — precomputed/cached)
- **3D MM-RoPE**: Positional encoding across `(t, h, w)` dimensions
- Native sparse attention (not in initial OSS release)
- No modality-specific attention/FFN — only in input/output layers + AdaLN branches

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
- `wubu_model_adapter.c`: Added `WUBU_ARCH_LFM25` detection (model_type="lfm2")
- `Makefile.win`: `src/wubu_dequant_nf4.o`, `src/wubu_h3_norm.o`, `src/wubu_mmrope.o` in `CORE_OBJ`

## 3D MM-RoPE Kernel

### H3-Omni-Transformer 3D Position Encoding
The H3-Omni-Transformer uses **3D Multimodal RoPE (MM-RoPE)** to encode positional
relationships across temporal and two spatial dimensions `(t, h, w)`:

- **head_dim** is split into 3 equal segments
- Each segment uses independent RoPE with its own base frequency (theta_t, theta_h, theta_w)
- **Temporal segment** (first 1/3): position varies by video frame
- **Spatial height segment** (middle 1/3): position varies by row
- **Spatial width segment** (last 1/3): position varies by column

### Implementation
- `include/wubu_mmrope.h` + `src/wubu_mmrope.c` — 3D MM-RoPE kernel
- Splits head_dim into 3 segments, applies standard rotary embedding per segment
- Each segment uses `cos(pos/theta^(2i/d))` and `sin(pos/theta^(2i/d))`
- 9/9 tests pass (init, validation, identity, reference match, NULL safety)

### Formula
```
For each segment s in {t, h, w}:
  seg_dim = head_dim / 3
  half = seg_dim / 2
  for i in [0, half):
    freq = pos_s / theta_s^(2i/seg_dim)
    x_i       = x_i * cos(freq) - x_{i+half} * sin(freq)
    x_{i+half} = x_i * sin(freq) + x_{i+half} * cos(freq)
```
- `gen_text_win`: Automatically picks up new objects via CORE_OBJ

## References
- bitsandbytes NF4: https://github.com/TimDettmers/bitsandbytes
- H3 paper: "H3: Hierarchical, Hypercomplex, Hypersymmetric"
- Inverse CDF table: computed via quantile function of N(0,1) at i/16 + 1/32
