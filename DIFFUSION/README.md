# Diffusion: Geometry-Guided Generative Models

The WuBu diffusion work applies hyperbolic geometry to generative modeling — replacing standard CNN U-Nets with Hyperbolic Geometric Attention (HGA) U-Nets.

## Contents

### hga-unet/
**`wubu_diffusion.py`** — The Hyperbolic Geometric Attention U-Net pipeline. The core script that proves the WuBu architectural philosophy works for generative tasks:
- Replaces standard convolution blocks with hyperbolic attention layers
- Built on JAX/Flax
- Trains without numerical explosion (a major milestone)
- The blueprint for applying WuBu philosophy to complex generative tasks

### funnel-diffusion/
The earlier diffusion experiments:
- **`WuBu_Funnel_Diffusionv0.1.py`** — Initial funnel diffusion attempt
- **`WuBu_Funnel_Diffusion_CLIP_VIDEOv0.1.py`** — CLIP-guided video diffusion
- **`WuBu_Total_Diffudion_V0.01.py`** — Full diffusion pipeline
- **`trainWuBuDiffusionINT_v0.01.py`** / **`generateWuBuDiffusionINT_v0.01.py`** — Training and generation scripts
- Training guides and documentation

### gan-vae-hybrid/
The GAAD (Geometric Adversarial Autoencoder Diffusion) family — hybrids of GANs, VAEs, and diffusion with geometric attention:
- Located in `draftPY/` (WuBuGAADHybridGen*.py, WuBuNestDiffusion*.py)
- Regional latent processing with optical flow (v0.10.1+)
- Motion vectors and temporal attention

## Key Insight

The HGA-UNet replaces the standard U-Net bottleneck with hyperbolic attention, which naturally captures the hierarchical structure of image features (coarse → fine). The geometry itself provides the multi-scale inductive bias — no need for explicit skip-connection engineering.
