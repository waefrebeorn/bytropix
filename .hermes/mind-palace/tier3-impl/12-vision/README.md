# Phase 5: Vision Port — WuBuVision

**Goal:** Port Qwen3.5+ vision encoder for multimodal capability.

**Depends on:** Phase 3 (text model trained and stable)
**Priority:** LOWEST — don't start until text model can generate coherent text.

## Target Vision Encoder (from Qwen3.6-35B-A3B config.json)

```
vision_config:
  depth: 27                    ← number of ViT layers
  hidden_size: 1152            ← vision token dimension
  intermediate_size: 4304      ← ViT MLP hidden dimension
  num_heads: 16
  patch_size: 16               ← spatial patch (pixels)
  temporal_patch_size: 2       ← temporal patch (frames)
  spatial_merge_size: 2        ← merge 2×2 spatial patches
  num_position_embeddings: 2304  ← max positions (2304 = max patches)
  out_hidden_size: 2048        ← projection to match text hidden
  hidden_act: gelu_pytorch_tanh  ← activation
  in_channels: 3               ← RGB
  deepstack_visual_indexes: [] ← empty (no deepstack)
```

**Major correction:** This vision encoder is NOT a standard ViT. It has:
- `temporal_patch_size: 2` — this is a 3D patch embedding (handles video)
- `spatial_merge_size: 2` — spatial downsampling after patch embed (2×2 → 1)
- `num_position_embeddings: 2304` — supports 2304 patches = for 1280×720×2 video

The vision encoder is closer to a **SigLIP-style 3D ViT** than a standard ViT.

## Architecture (from clues in config + weights)

**This phase is blocked until we can dump the vision encoder weights from the GGUF.**
The vision encoder weights are stored in the same GGUF file, but we haven't extracted them.
They'll have tensor names like `v.encoder.layer.0.*` or similar.

For now, the approximate structure:
```
Image [3, T, H, W]    (T=1 for single image, T=2+ for video)
    ↓
3D Conv (patch=16, temporal=2) → [N_spatial, N_temporal, 1152]
    ↓
spatial_merge_size=2 → downsample spatial 2×2 → [N_spatial/4, N_temporal, 1152]
    ↓
+ Position embeddings (3D: [H_grid, W_grid, T])
    ↓
27× ViT layers (standard: MSHA + MLP with GELU)
    ↓
Projection: 1152 → 2048 (out_hidden_size)
    ↓
MRoPE positions prepended for text model interleaving
```

### Vision Encoder Weight Extraction (TODO when Phase 5 starts)

Run the GGUF dumper to find vision tensor names:
```
TENSOR[N] v.patch_embed.weight     [1152, 3, 16, 2] maybe?
TENSOR[N] v.pos_embed              [2304, 1152] maybe?
TENSOR[N] v.encoder.layer.0.*      ViT weights
...
TENSOR[N] v.projection.weight      [1152, 2048]
```

## MRoPE Details (from text config)

The MRoPE (Multi-Resolution RoPE) is defined in the text model's `rope_parameters`:
```json
"rope_parameters": {
    "mrope_section": [11, 11, 10],    // 32 total dims: 11 for H, 11 for W, 10 for T
    "mrope_interleaved": true,
    "rope_type": "default",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25
}
```

This means RoPE is applied to 64 of 256 head_dim dimensions (partial=0.25),
and those 64 dims are split 11/11/10 for H/W/T positions with 32 dims interleaved.
The remaining 32 dims of the 64 are... actually: partial=0.25 on head_dim=256 = 64 dims rotated.
mrope_section sums to 32. So 32 of the 64 rotated dims are MRoPE, the other 32?
Maybe mrope_section only applies to the linear attention QK heads (head_dim=128, 0.25×128=32).

**This needs verification from llama.cpp source.**

## WuBuVision Grafting

When text model is stable:
1. Port the 3D ViT to C (straightforward — standard ViT layers)
2. Load vision weights from GGUF
3. Add hyperbolic output: 1152 → exp_map → 2048 via learned projection
4. MRoPE integration with text tokens
5. Train vision-language alignment (image captioning data)

## Files to Create
```
src/wubu_vision.c              — 3D ViT vision encoder
include/wubu_vision.h          — Header
src/wubu_mrope.c               — Multi-resolution RoPE
include/wubu_mrope.h           — Header
```

## Pitfalls

1. **Vision encoder is a 3D ViT, not 2D.** `temporal_patch_size=2` means the first
   conv layer consumes 2 video frames simultaneously. For single images, we need to
   either stack 2 copies or use a different conv weight.

2. **Spatial merge is non-standard.** After patch embedding (16×16 patches), spatial_merge=2
   merges 2×2 adjacent patches into one. This means a 256×256 image (16×16=256 patches)
   becomes 8×8=64 merged patches. The position encoding is applied AFTER this merge.

3. **MRoPE implementation is undocumented.** The only reference is in Qwen3's config.json.
   We'll need to read llama.cpp's `qwen35` model code to understand the exact implementation.

4. **Vision weights are in a different format.** The vision encoder might use F32 or F16
   weights (not quantized like the text model), because vision encoders are typically
   smaller and more sensitive to quantization.
