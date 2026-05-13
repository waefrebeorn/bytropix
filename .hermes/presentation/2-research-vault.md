# Research Vault Tour

**Purpose:** A guided walkthrough of the bytropix research vault (`../vault/`) — what each area explored, what was learned, and how results feed into the current WuBu project.

---

## theory/

Examines the geometric philosophy underpinning WuBu. Physics papers and axiomatic derivations (e.g., `02-axiomatic-emergent-theory.md`) argue that nested hyperbolic spaces with learnable curvature — rather than brute-force Euclidean scaling — are the correct geometry for representational learning. The central formalism `Q = Σ q_k ∏ α_i^E` appears throughout. `../vault/theory/README.md`

## encoders/

Six research phases of encoder design in a single folder. Phase 1 (symmetric geometric AE) established U-Net-style geometric skip connections and `.wubu` native latent format. Phase 2 (topological QAE) compressed images to as few as 3 Hamiltonian floats — extreme but fragile. Phase 3 built a VQ-VAE + hierarchical conductor transformer for text-to-image generation (576 tokens/image), supported by the 66K-line CORPUS.py training corpus. `../vault/encoders/README.md`

## hash-mind (encoders/)

WuBuMind V1–V7.1 in JAX/Flax: the full progression from hyperbolic kNN attention (V1) through dual-agent Q-learning (V5) to a complete BPE-tokenized training pipeline with geometric embedding and BallTree indexing (V7.1). Also includes SimpleHash V1–V3 (rolling hash attention, the direct precursor to the C port). `../vault/hash-mind/README.md`

## hamilton (encoders/)

30+ Python files from a month of geodesic encoder experiments (Nov 2025). The flagship `Wubu_Monolith.py` and `chimera_quaternion.py` established the quaternion attention concept that later became the CUDA Hamilton kernel. Validated "Energy-Based Manifold Learning" as a workable approach (Nov 22 commit). `../vault/hamilton/README.md`

## attention/

Four attention variants studied. **Sparse attention** (O(n·k) dual memory) is the most viable for C/CUDA porting with clean PyTorch ops. **Topological sequence model** (Conv1D compression, O(n) complexity) inspired the Hamilton CUDA kernel. **Hyperbolic attention** ("Tri-Cameral Mind") proved pedagogical but not practical — low port viability. **Entropix sampler** is an inference-time sampling strategy (not attention) with heavy jax.scipy.special dependencies that don't translate to CUDA natively. `../vault/attention/README.md`

## audio/

WubuSynth galactic core synthesizer — an unsupervised adversarial audio pipeline using EnCodec tokenization with harmonic enhancement and VHF radio chain processing. Demonstrates the WuBu geometric approach extends beyond text/image into the audio domain. `../vault/audio/README.md`

## diffusion/

Hyperbolic Geometric Attention UNet (hga-unet) as a diffusion backbone, plus funnel diffusion experiments with KL-divergence guidance, CLIP-conditioned video diffusion, and GAN/VAE hybrids for image-space conditioning. Geometric inductive biases were applied to the denoising process rather than to latent encoding. `../vault/diffusion/README.md`

## optimizers/

Q-learning-based LR controller (10-state × 5-action Q-table with ε-greedy exploration) and a PID Lambda Controller for second-order loss balancing. Each clone in the multi-agent "Project Chimera" architecture received its own PID agent managing loss weights as control signals. These meta-learning approaches were experimental and did not carry forward into the current C/CUDA training loop. `../vault/optimizers/README.md`

## c-training/

Pure C port of the hash-mind rolling hash attention transformer — 210K parameters, 4-layer, d_model=64, trained from scratch with manual backprop, no autograd. Achieved ~4000 steps/sec on 50K chars of Shakespeare (CPU, -O3). Established the porting pattern (JAX → C → CUDA) and exposed the need for gradient clipping at 0.5 and careful LR tuning (0.0003). The CUDA kernels here (`ggml-cuda/wubu-cuda.cu`) implement Poincaré exponential map and MoE routing. `../vault/c-training/README.md`

## lean-proofs/

Formal verification of WuBu math in Lean 4. Four proof files tackle: Poincaré ball exp/log identities (partial), Möbius addition preserving the ball (1D proven), KV compression error bound (sketch), and hyperbolic gyration preserving the ball (sketch). These are early-stage — the mathlib4 build is still compiling — but establish the intent to make the WuBu geometry formally grounded rather than empirically guessed. `../vault/lean-proofs/README.md`

---

**Note:** This vault represents exploratory research from Aug 2025 onward. Results vary: some areas (hash-mind rolling attention, Hamilton quaternion encoders) directly informed the current C/CUDA implementation; others (diffusion, audio, optimizers) remain as standalone experiments that may be revisited in later phases.
