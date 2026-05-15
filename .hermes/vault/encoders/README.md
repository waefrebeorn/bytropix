# Vault: ENCODERS — Research Core (5 Phases)

## Phase 1: Symmetric Geometric Autoencoder
`symmetric_geometric_autoencoder.py` — Flagship AE with U-Net-style geometric skips. Learns manifold image representation. Output: `.wubu` native latent format. `compressor.py` compresses images into tiny files.
Key: `wubumind_codec.py`, `manipulator_v2.py` (latent style transfer)

## Phase 2: Topological Autoencoders (QAE)
`qae.py` — Holomorphic Quantum Autoencoder. Compresses images to 3 floats (Hamiltonian coefficients). "Quantum Observer" reconstructs by evolving quantum system.
Key: `QAE2.py`, `quant.py`/`quant2.py`/`quant3.py` (extreme compression experiments)

## Phase 3: Generative (Text-to-Image)
`phase3_generative.py` — VQ-VAE tokenizer → hierarchical conductor transformer (Kid/Student/Polisher). CLIP text conditioning. `backupcorpus.py`/`CORPUS.py` (14.6MB, 66K lines) — 18 LORE dictionaries of training data. `FORXLADEVS.py` — standalone WubuMind text-generation.
Key: 3-stage pipeline: AE latent → VQ codes → transformer generation

## hash-mind (Core JAX Portfolio)
WuBuMind versions V1-V7.1 (JAX/Flax).
- `WuBuMindV7.1.py` — Full training pipeline: tokenizer → navigator → oracle → funnel cake → generate
- `WuBuMindJAX.py` — Hyperbolic kNN attention + toroidal gradient descent
- `WuBuMindJAXv3CORPUSPASTE.py` — CORPUS-trained, dual-agent Q-learning
- `WuBuNest_Trainer.py` — Full JAX training with DDP, AdamW
- `WuBuNestmRnaTrainer.py` — mRna variant (PyTorch, no Triton)
- `SimpleHashV1-V3.py` — Rolling hash attention (precursor to C port)

## hamilton-encoder-cpu (Geodesic AI Brain)
30+ Python files — a month of experiments (Nov 2025).
- `Wubu_Monolith.py` — Flagship geodesic encoder
- `chimera_quaternion.py` — Quaternion attention
- `Wubu_Geodesic_Sphere.py` — Geodesic rendering
- `Wubu_Geodesic_Validation_Suite.py` — Stress testing
- `Wubu_Physics_Verification_Suite.py` — Physics compliance
Key insight: validated "Energy-Based Manifold Learning" (Nov 22 commit)

---

*Part of the WuBuText AI project. See [Project Overview](../../README.md) for navigation.*
