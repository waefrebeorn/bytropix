# 043 — The individuals, deep-dived: Tri Dao, Zhiqi Bu, and the console underground

Status: written 2026-08-03. Corpus: the full published record (scholar
profiles, blogs, papers, videos, transcripts archived in the wubuos
compendium `05-sources/`). Purpose: the OPT avenue (1000 gaps, wubuos
`04-roadmap/opt-bank.md`) derives its real mechanisms from these people.
Credit every individual; every gap is traceable to a published result.

## 1. Tri Dao (Princeton, Dao-AILab) -- the kernel-systems lineage

The full corpus (Google Scholar, citations in parens): FlashAttention
(2022, 5619), FlashAttention-2 (2023, 3267), FlashAttention-3 (2024,
636, with Shah/Bikshandi/Zhang/Thakkar/Ramani), Mamba (2023, 12074,
with Albert Gu), Transformers-are-SSMs (2024, 2039, with Gu), HIPPO
(2020, 1287, with Gu/Ermon/Rudra/Re), S4nd (2022, 395), Hungry Hungry
Hippos (2023, 986), Hyena (2023, 675, with Poli/Massaroli/...), Medusa
(2024, 818, with Cai/Li/Geng/...), Deja vu (2023, 545), Scatterbrain
(2021, 226, with Chen), Caduceus (2024, 278), the kernel theory of data
augmentation (2019, 296), the empirical Mamba study (2024, 243),
decentralized training (2022, 183, with Yuan). The 2026 Gram-NS blog
(with Jack Zhang, Noah Amsel, Berlin Chen) + the Quack symmetric-GEMM
library.

The converged principles (what the U-Bus took):
- **IO-awareness**: the attention kernel's bottleneck is the HBM
  traffic, not the FLOPs -- the tile (the SRAM-resident block) is the
  unit of work (OPT-C tiling theme).
- **The square-space optimizer**: the Gram iteration drops the
  rectangular FLOPs ~5x (OPT-D, IMPLEMENTED as gpu_wubu_ns5_gram --
  committed acdf13d).
- **The symmetric GEMM**: the A=MM^T half is redundant work (OPT-D03,
  the future kernel).
- **The composed R**: the NS polynomial composition as ONE final
  rectangular GEMM.
- **The square-case-luck trap**: a square-only probe hides the
  row-major/col-major mappings (OPT-D04 -- the test now covers
  square/wide/tall/small).
- **The stream of work**: FlashAttention-3's asynchrony + low
  precision (the OPT-I command-list theme).
- **The sparse/decomposed attention** (Deja vu, Scatterbrain): the
  contextual sparsity + the low-rank splits (the OPT-J packing theme).

## 2. Zhiqi Bu (Meta FAIR, Superintelligence Labs) -- the growth lineage

The full corpus: the deep progressive training paper ("On the Optimal
Depth of Neural Networks..."), fastDP (the most efficient/scalable DP
library -- 100B LLM+vision), UPQ (unified progressive quantization to
2-bit instruction-tuned LLMs), DP-Adam/DP-LayerNorm, the function-
preserving growth family, distributed learning, the DP theory.

The converged principles (OPT-E theme):
- **zero/one-layer progressive training**: start shallow, grow deep --
  ~5x compute savings at equal loss.
- **mixing-time transfer**: the expansion happens at tau ~ 0.8T; the
  grown model needs DATA, not iterations, to catch up.
- **function-preserving insertion**: the new layer is the identity/
  zero-gated at birth -- the loss must NOT jump (the DA validation).
- **the 60x depth scaling** on GPT-2-scale models.
- **progressive quantization**: the precision grows with the stage
  (UPQ -- 2-bit weight-only, the OPT-J packing theme).
- **the growth operator**: observe -> decide -> mutate -> validate --
  the amoeba loop's exact shape.

## 3. The console underground (the full wave) -- the machine lineage

- **Kaze Emanuar + Silas Lock**: the folded polynomial, the sine-table
  mis-optimization, the compute-vs-fetch roofline (research/042).
- **Rodrigo Copetti**: the N64 architecture analysis (the RDRAM
  640ns latency, the 9-bit bus, the RSP/RDP split, the RCP task
  scheduler).
- **The n64decomp scene** (Kenix, Rozlette, decomp.me, ZRET): the
  decomp-as-method -- rebuild the binary as readable C, the parity
  test as the oracle (OPT-F).
- **TonC/coranac (Jasper Vijn)**: the GBA IWRAM/EWRAM hierarchy, the
  DMA-vs-loop measurements (~10% faster), the THUMB/ARM instruction
  split (OPT-H).
- **The PS1 developers**: the GTE fixed-function, the affine-texture
  subdivision (OPT-G).
- **The Dreamcast/PowerVR lineage**: the tile-based deferred renderer,
  the 32x32 on-chip tile (OPT-C).
- **The demoscene**: procedural generation, the 64K intros, the
  trade-memory-for-compute discipline (OPT-J).
- **The GTA:VC Dreamcast port team**: the 8MB packing of a full game.
- **John Carmack / id**: the fast inverse sqrt -- the exponent fold
  (OPT-A).

## 4. The convergence, one line each

- **OPT-A (folded math)**: every starved machine folds the domain
  until the polynomial is tiny (Kaze's [0,pi/4], Carmack's exponent
  bit-cast, the DSP fixed-point).
- **OPT-B (streaming)**: the bus is a stream; latency is the enemy;
  data movement >> compute (Kaze's audio, the RDRAM, the weight
  cache).
- **OPT-C (tiling)**: the tile is the on-chip unit of work (PowerVR,
  FlashAttention, the GBA fast tier).
- **OPT-D (optimizer math)**: iterate where the matrix is SQUARE (the
  Gram) -- the rectangular work is the cost (Tri Dao).
- **OPT-E (growth)**: grow depth, not width, at the mixing time, with
  function-preserving insertions (Zhiqi Bu).
- **OPT-F (decomp method)**: the source is the oracle; parity is the
  proof (the n64decomp scene).
- **OPT-G (capability adaptation)**: subdivide the data until it fits
  the fixed-function unit (PS1).
- **OPT-H (SIMD)**: branchless, inline-header, the vector floor
  (coranac, the SVML).
- **OPT-I (command lists)**: the engine executes a program of ops
  (the RSP display list, the PS1 FIFO, the CUDA graphs).
- **OPT-J (packing)**: generate, don't store; pack at the true bit
  width (the demoscene, the 9-bit bus, UPQ).

## 5. The ledger

- research/042: the Kaze/Silas deep dive + the 7-hop.
- The OPT avenue: 1000 gaps (opt-bank.md), 9 wired: the Gram-NS (the
  square-space iteration, ~2.3G vs 4.9G MACs), the trace-normalization
  fix (whole-matrix), the foldmath (Silas/Kaze, 1.5e-7 on the RoPE
  range), the weight cache, the head-grad GEMMs, the FD as the oracle.
- The transcripts: compendium/05-sources/kaze-*.md.
- Every gap in the bank carries a driver-tag to its individual.
