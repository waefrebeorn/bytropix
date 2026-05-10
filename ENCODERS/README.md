# ENCODERS/ — The WuBu Nesting Research Core

This is it. The meat. The laboratory notebooks of the WuBu Nesting project, organized by research phase from earliest to most experimental. Everything here is a raw research script — unpolished, often unfinished, sometimes contradictory — but each file represents a real intellectual bet, a hypothesis tested in JAX, Flax, and pure obsession.

If the rest of this repo is the cathedral, this directory is the scaffolding, the sketches, and the coffee-stained blueprints.

---

## The Research Arc (Five Phases)

### Phase 1: Symmetric Geometric Autoencoder
**Folder:** `phase1-symmetric-encoder/`

The starting point. The core insight that launched everything: what if you could learn a true *geometric* manifold for image data, then use a symmetrically-designed decoder to unfold it back?

Key files:
- **`symmetric_geometric_autoencoder.py`** — The flagship. A symmetric autoencoder with U-Net-style geometric skip connections. Learns the fundamental manifold of image data without diffusion.
- **`compressor.py`** — Compresses images into `.wubu` files (the project's native latent format). Tiny files, big ideas.
- **`manipulator.py` / `manipulator_v2.py`** — Style transfer and region editing *in latent space*. Not pixel-pasting — semantic manipulation in the model's own language.
- **`wubumind_codec.py`** — The codec layer that bridges latent representations and the real world.
- **`structure_generator.py`** — Generates structures from the learned manifold.
- **`prepare_manifold_data.py` / `realign_manifold_data.py`** — Data prep pipeline for manifold learning.

This phase proved that geometric latent spaces could be used for meaningful compression and semantic editing. The `.wubu` format was born here.

---

### Phase 2: Topological Autoencoders (QAE)
**Folder:** `phase2-topological-ae/`

The wild pivot. "What if the latent space isn't just geometric — what if it's *quantum mechanical*?"

Key files:
- **`qae.py`** — The Holomorphic Quantum Autoencoder. Compresses images into just **3 floating-point numbers** — coefficients of a Hamiltonian that represents the entire image. The "Quantum Observer" reconstructs the image by evolving a quantum system.
- **`QAE2.py`, `QAE_Advanced.py`** — Iterations on the quantum autoencoder concept, adding complexity and capability.
- **`quant.py`, `quant2.py`, `quant3.py`** — Quantization experiments. Pushing latent representations toward extreme compression.
- **`topological_ae_trainer.py`** — Training infrastructure for topological autoencoders.

The quantum autoencoder phase was the first hint that extreme compression ratios were achievable — not through brute-force bit packing, but through representing information in the *parameters of the laws that generate it*.

---

### Phase 3: Generative Models
**Folder:** `phase3-generative/`

Text-to-image enters the chat. Phase 3 bridges the geometric encoder from Phase 1 with transformer-based generative modeling.

Key files:
- **`phase3_generative.py`** — The main event. Full text-to-image pipeline: paired data preparation, VQ-VAE tokenizer training, transformer "Conductor" training, and generation/inference.
- **`Phase3_maybetext.py`, `Phase3FAT.py`, `phase3_headscrathc.py`, `phase3_racinghtebeam.py`, `phase3_timeanddepth.py`** — Experimental variations. Trying different architectures, loss functions, and training strategies.
- **`tokenizer_training.py`** — Training the discrete tokenizer (visual vocabulary) from the continuous latent space.
- **`poem_dataset_generator.py`** — Synthetic data generation for testing.
- **`corpus_builder.py`, `corpus_converter.py`, `commentify.py`** — Text corpus tooling.

This phase proved the geometric latent space could *feed* a generative transformer, producing images from text prompts.

---

### Phase 4: HashMind
**Folder:** `hash-mind/`

The philosophical leap. "What if we don't train a neural network at all? What if we *hash* the data into a geometric memory structure and retrieve by association?"

HashMind is the project's attempt at a completely different paradigm: non-differentiable, attention-free, hash-based associative memory. It treats language as geometry.

Key files:
- **`SimpleHashV1.py` through `SimpleHashV3.py`** — The earliest hash-based prototypes.
- **`WuBuMindV1.py` through `WuBuMindV7.1.py`** — The evolution of the WuBuMind architecture. Each version tried different hashing strategies, manifolds, and retrieval mechanisms.
- **`WuBuMindJAX.py`, `WuBuMindJAXv2.py`, `WuBuMindJAXv3CORPUSPASTE.py`, `WuBuMindJAXv5.py`, `WuBuMindJAXv1337.py`** — JAX-accelerated versions of the HashMind concept.
- **`wubu_nesting_impl.py`, `wubu_nesting_example.py`, `wubu_nesting_visualization.py`** — The "nesting" concept: hierarchical hash structures that capture multi-scale semantic patterns.
- **`wubuMindv4WEBRADIO.py`, `wubuMindv4JAX.py`** — Specialized variants.
- **`wubumind_galactic_core_v1.py`, `wubumind_galactic_core_v3_qlearn.py`** — The "Galactic" line: scaling HashMind to planetary-scale datasets with Q-learning controllers.
- **`WuBuNest_Trainer.py`, `WuBuNest_Inference.py`, `WuBuNestmRnaTrainer.py`** — The WuBuNest training/inference pipeline.
- **`ProjectAgentChimera.md`, `GAAD-WuBu-ST*.md`** — Design documents for multi-agent training systems (Chimera architecture).

HashMind represents the most radical departure from mainstream ML in this project. It asks: "What does learning look like when you remove backpropagation entirely?"

---

### Phase 5: Hamilton Encoder (CPU / Geodesic Layers)
**Folder:** `hamilton-encoder-cpu/`

The return to physics. "If the manifold is geometric, and the compression is extreme, then *how do the geodesics actually work?*"

This phase is a deep dive into the mathematical and physical foundations of the encoding process itself. It's the most theoretically ambitious phase of the project.

Key files:
- **`Wubu_Geodesic_Layer_v1.py` through `Wubu_Geodesic_Layer_Final.py`** — The geodesic layer series. Multiple implementations of learnable geodesic pathways through latent space.
- **`Wubu_Geodesic_Sphere.py`, `Wubu_Geodesic_StressTests.py`, `Wubu_Geodesic_Benchmarks.py`, `Wubu_Geodesic_Benchmark_Pro.py`** — Testing, validation, and benchmarking of geodesic computations.
- **`Wubu_Geodesic_Validated.py`, `Wubu_Geodesic_Validation_Suite.py`, `Wubu_Physics_Verification_Suite.py`** — Formal verification of geometric correctness.
- **`Wubu_Geodesic_Hybrid.py`, `Wubu_Geodesic_Hybrid_Fixed.py`, `Wubu_Geodesic_Lossless.py`, `Wubu_Geodesic_Active_Recall.py`** — Extensions: hybrid architectures, lossless encoding, active recall mechanisms.
- **`Wubu_Geodesic_Storage.py`** — Storage layer for geodesic data structures.
- **`Wubu_Complex_Field.py`, `Wubu_Neural_Field.py`, `Wubu_Spectral_Field.py`** — Field-theoretic approaches to encoding.
- **`Wubu_Monolith.py`, `Wubu_Quaternion_Monolith.py`, `Wubu_Hex_Lite.py`, `Wubu_VGA_Ultimate.py`** — Integrated architectures combining multiple ideas.
- **`Wubu_TriCameral_Mind.py`, `Wubu_Crucible_Experiment.py`, `Wubu_Orbital_Decay_Experiment.py`, `Wubu_Island_Boat.py`** — Experimental architectures pushing the boundaries.
- **`chimera_Resnet.py`, `chimera_quaternion.py`** — Chimera architecture implementations combining ResNet and quaternion dynamics.
- **`entropiximagepatch.py`** — Entropy-based image patching strategies.
- **`gptcanvas.py`, `gptphysics.py`, `gpt_at_home.py`, `coco_preprocessor.py`, `annotations_db.py`** — Support tooling for data processing and experimental interfaces.
- **`WuBuTheory.MD`** — The formal theoretical paper: "The Axiomatic-Emergent Theory of Physical Law." This is the physical theory that underlies the entire encoding philosophy.
- **`prior_art.MD`** — Survey of related work.
- **`RESNET.md`** — Documentation for the ResNet-based Chimera architecture.

This phase is active exploration. The geodesic layers are the mathematical engine that the earlier phases assumed existed but hadn't fully built.

---

## What This Is (And Isn't)

These scripts are **research artifacts**, not production code. They are:

- ✅ **Idea-rich** — Each file explores a real conceptual bet
- ✅ **Working demos** — Many are runnable (see their individual docs)
- ✅ **Educationally valuable** — Read them to understand the progression of ideas
- ❌ **Not polished** — Expect rough edges, debug prints, and half-finished refactors
- ❌ **Not a library** — There's no unified API; each phase has its own conventions
- ❌ **Not guaranteed to run** — Dependencies change, checkpoints get lost

The value is in the **ideas**, not the code quality. Read these like a scientist reads lab notebooks.

---

## CUDA / Production Integration

For production-optimized CUDA implementations of the core encoding ideas, see:

**`LLAMA-CPP-INTEGRATION/`** (at the repo root)

That directory contains the C/CUDA ports of the key encoding layers, integrated with the llama.cpp inference stack for real-world deployment. The research here in `ENCODERS/` is the proof-of-concept; the CUDA integration is the product.

---

## Getting Started

Want to explore? Each phase folder has its own documentation:

| Phase | Key Doc | What To Run First |
|-------|---------|-------------------|
| Phase 1 | `phase1_encoder_update.md` | `symmetric_geometric_autoencoder.py train` |
| Phase 2 | `RUNQAE.md` | `qae.py train` |
| Phase 3 | `phase3.md` | `phase3_generative.py train-conductor` |
| Phase 4 | `RUNNING WuBuMindv7.1.md` | `WuBuMindV7.1.py train_navigator` |
| Phase 5 | `RESNET.md`, `WuBuTheory.MD` | `Wubu_Geodesic_Layer_Final.py` |

---

> **"The universe is not arbitrary but axiomatic, built upon a minimal set of logical rules whose consequences we observe as physical law."**
> — from *WuBuTheory.MD*, the paper that ties it all together

Enjoy the rabbit hole. It goes deep.
