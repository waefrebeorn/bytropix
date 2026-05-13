# Vault: Diffusion — Hyperbolic Geometric Attention for Generation
#
## Location: `DIFFUSION/`

### hga-unet/
Hyperbolic Geometric Attention UNet — diffusion backbone with geometric inductive biases.

### funnel-diffusion/
- `WuBu_Funnel_Diffusionv0.1.py` — Funnel diffusion with KL-divergence guidance
- `WuBu_Funnel_Diffusion_CLIP_VIDEOv0.1.py` — CLIP-conditioned video diffusion
- `WuBu_Total_Diffudion_V0.01.py` — Early exploration
- `trainWuBuDiffusionINT_v0.01.py` / `generateWuBuDiffusionINT_v0.01.py` — Training + inference
mple.py` / `sync_by_trimming.py` — Video processing utilities
    14|- `infer.py` — Inference entry point
    15|
    16|### gan-vae-hybrid/
    17|GAN/VAE hybrid for image space conditioning.
    18|

---

*Part of the WuBuText AI project. See [Project Overview](../../README.md) and [Presentation Layer](../presentation/README.md) for navigation.*
