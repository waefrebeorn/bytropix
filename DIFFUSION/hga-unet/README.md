# HGA-UNet (Hyperbolic Geometric Attention U-Net)

This directory documents the HGA-UNet architecture referenced in the WuBu philosophy document:

> *"`wubu_diffusion.py` (HGA-UNet): A functional pipeline for a diffusion model built on a pure Hyperbolic U-Net, replacing standard CNNs with geometric attention."*

The reference implementation was described conceptually in PHILOSOPHY.md. The actual diffusion experiments are in:
- `DIFFUSION/funnel-diffusion/` — Funnel diffusion and CLIP video diffusion
- `ENCODERS/hash-mind/` — The geometric attention building blocks
- `draftPY/` — GAAD hybrid diffusion variants (WuBuGAADHybridGen*.py, WuBuNestDiffusion*.py)

The HGA-UNet would replace the U-Net bottleneck with WuBu hyperbolic attention layers using the implementation patterns in `wubu_nesting_impl.py`.
