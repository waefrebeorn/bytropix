# Moondream3 Vision Manifold Integration (Phase 5b)

**Secondary vision encoder path** — comes after Qwen 3D ViT (Phase 5a) is stable.

Moondream3's vision encoder (SigLIP-style, ~300M params) gets extracted via vLLM
and ported into C as direct native tensors — same pattern as the Qwen GGUF rip.

No HTTP proxy. No external service. The weights get dumped to binary and loaded
directly in the WuBu forward pass, producing tokens in the Poincaré ball.

---

## Approach: vLLM Weight Dump → C Port

Same pipeline as Phase 0-1 (GGUF embed rip):

```
Moondream3 safetensors (45G)
    │
    ▼  vLLM loads the model
    │
    ▼  Python dump script extracts vision encoder weights
    │    v.encoder.layer.*, v.patch_embed.*, v.projection.*
    │
    ▼  Binary weight files (f32)
    │    data/moondream3_vision_weights.bin
    │
    ▼  C port: wubu_vision_moondream.c loads weights
    │    Runs SigLIP-style ViT forward pass
    │
    ▼  Hyperbolic graft
    │    exp_map(vision_features) → Poincaré ball
    │
    ▼  Concatenate vision tokens + text tokens in same hyperbolic space
```

## Manifold Integration

```
                  ┌─────────────────────────────┐
                  │     WuBu Hyperbolic Manifold │
                  │   (Poincaré ball, R=0.956)   │
                  └──────────┬──────────────────┘
                             │
             ┌───────────────┼───────────────┐
             │               │               │
             ▼               ▼               ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐
       │  Text    │   │ Vision   │   │ Vision   │
       │ Tokens   │   │ (Moondr3)│   │ (QwenVT) │
       │ (native) │   │ (Phase5b)│   │ (Phase5a)│
       └──────────┘   └──────────┘   └──────────┘
```

All three paths produce tokens in the same Poincaré ball.

---

## Implementation Phases

### Phase 5b.1: Weights Dump from vLLM

Use vLLM to load moondream3-preview, iterate its vision encoder layers,
dump weights to binary files.

```python
# tools/dump_vision_weights.py — scratch tool (same job as dump_gguf.py)
import torch
from vllm import LLM

model = LLM(model="moondream/moondream3-preview", ...)
# Access model.llm_engine.model_executor...
# Dump v.patch_embed, v.encoder.layer.*, v.projection.*
# Write to data/moondream3_vision_weights.bin (f32)
```

**Deliverable:**
- `tools/dump_moondream3_weights.py` — weight extraction script
- `data/moondream3_vision_weights.bin` — extracted f32 weights
- `data/moondream3_vision_config.txt` — architecture params (patch_size, depth, hidden, etc.)

**Verification:**
- Weight shapes match expected SigLIP architecture
- Forward through dumped weights in Python matches original forward

### Phase 5b.2: C Vision Encoder Port

Write SigLIP-style ViT in C, load the dumped weights.

**Files:**
```
src/wubu_vision_moondream.c    — ViT forward: patch embed → 27 layers → projection
include/wubu_vision_moondream.h
tools/test_vision_moondream.c  — Verify C output matches Python reference
```

**ViT layers to port:**
1. Patch embedding (conv2d, 14×14 patches)
2. Position embeddings (learned 1D pos)
3. Pre-LN transformer layers (MSHA + MLP, GELU)
4. Output projection (1152→2048) → exp_map → Poincaré

**Verification:**
- C forward output ±1e-4 of Python reference on test images
- Vision token norms in Poincaré ball ≈ R/2 (matching text tokens)

### Phase 5b.3: Hyperbolic Vision Training

Train WuBu's attention layers to understand vision tokens.

- Freeze vision encoder (dumped weights are static)
- Train only the text model's cross-attention to vision tokens
- Data: image-caption pairs via the C vision encoder

### Phase 5b.4: Full Integration (optional)

If useful, fuse the vision encoder into the same GGUF file format
used by the rest of the model, enabling unified weight loading.

---

## Moondream3 Vision Encoder Spec (from huggingface config)

- Type: SigLIP-style ViT
- Depth: 27 layers
- Hidden: 1152
- Intermediate: 4304
- Heads: 16
- Patch size: 14×14
- Image size: 448×448
- Output dim: 2048 (projection matches text hidden)
- Activation: GELU (approximate tanh)

---

## Comparison: Moondream3 vs Qwen 3D ViT

| Feature | Moondream3 Vision | Qwen 3D ViT |
|---------|-------------------|-------------|
| Encoder type | SigLIP ViT (standard 2D) | 3D ViT (temporal/spatial merge) |
| Encoder params | ~300M | ~1.5B |
| Output dim | 1152 → project to 2048 | Same |
| Patch size | 14×14 | 16×16, temporal=2 |
| Image resolution | 448×448 (fixed) | Flexible, up to 1280×720 |
| Video support | Frame-by-frame | Native temporal patches |
| Weight source | vLLM dump → .bin | GGUF extract |
| C port pattern | Same as Phase 0-1 (GGUF rip) | Same |
| Priority | **Phase 5b** (after 5a) | **Phase 5a** (first) |

---

## Roadmap Status

| Sub-phase | Component | Status | Deliverable |
|-----------|-----------|--------|-------------|
| **5b.1** | Weight dump script | ⏳ TODO | `tools/dump_moondream3_weights.py` |
| **5b.1** | Dumped weight files | ⏳ TODO | `data/moondream3_vision_weights.bin` |
| **5b.2** | C ViT forward | ⏳ TODO | `src/wubu_vision_moondream.c` |
| **5b.2** | Vision test harness | ⏳ TODO | `tools/test_vision_moondream.c` |
| **5b.2** | Poincaré graft (exp_map) | ⏳ TODO | Include `wubu_mobius.h` |
| **5b.3** | Vision-language training | ⏳ FUTURE | Joint training loop |

---

## References

- Moondream3 cached: `~/.cache/huggingface/models--moondream--moondream3-preview/`
- GGUF rip pattern (same approach): `tools/dump_gguf.py`
- WuBu Poincaré math: `THEORY/` + `include/wubu_mobius.h`
- Qwen ViT plan (Phase 5a): `./README.md`
