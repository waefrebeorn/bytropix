# The U-Bus: an AGI LLM substrate designed like an N64 RAM bus

Status: DESIGN v1 (2026-08-03) + `src/wubu_ubus.c` (the substrate, real + tested)
Audience: the wubuwizard engine. Doctrine: no stubs, tests != correct, measure.

## 0. Why the N64

The N64 was the last console built on a genuinely *unified* memory philosophy:

| N64 | AGI LLM analogue |
|---|---|
| RDRAM (4-8MB) + cartridge ROM in ONE address space | one flat pool: weights (cartridge) + KV/activations (RDRAM) + optimizer state |
| the 562 MB/s bus is THE constraint (famously) | data movement is the constraint (roofline: 2-17% of wall time is even the optimizer's math) |
| cartridge = read-mostly ROM streamed at full bus width | the checkpoint = a read-mostly cartridge streamed to whatever compute is attached |
| RSP: a programmable vector co-processor (geometry) | the GEMM-class engine (GPU/NPU today, anything registered tomorrow) |
| RDP: fixed-function rasterizer | the fixed-function engine slots (attention, norms, NS5, CE) |
| swap/expand the cartridge = new game | grow the cartridge = progressive depth/width expansion, function-preserving |

The rule that falls out: **the bus is designed first, the engines are fungible
behind it.** Every op declares its byte-budget; the cost model picks the
cheapest physical path. That is what makes the design *agnostic* — the
selector is a pure function of a measured capability table, not a vendor.

## 1. The measured hardware truth (this box, 2026-08-03)

| resource | measured | how |
|---|---|---|
| GPU fp32 SGEMM (RTX 4050 Laptop, sm_89, 6GB) | **1884.7 GFLOPS** | cuBLAS 2048^3 bench |
| GPU fp16 tensor-class | ~194 TFLOP/s (spec; fp16 path pending) | — |
| CPU (Ryzen 7 7445HS, 12 threads, AVX-512) | ~100-200 GFLOPS fp32 | Zen4 spec |
| PCIe host<->device | ~10-16 GB/s | laptop x8 Gen4 |
| DRAM (DDR5, dual channel) | ~30-50 GB/s | Zen4 laptop |
| VRAM (GDDR6) | ~256 GB/s | 4050 class |
| storage SSD | 712 GB free at /home/wubu | — |
| SD card | drvfs-slow: DOWNLOADS ONLY | hard rule |

The training step (seq 128, real corpus): ~35 GFLOP/step -> the GPU does the
math in ~20ms. We measure ~350ms fwd+bwd. The gap is *not compute*: it is
(1) the CPU loops still on the critical path (attention backward, softmax),
(2) per-call H2D uploads (killed by the weight cache, gen-invalidated),
(3) the optimizer (killed by the GPU NS5). The U-Bus makes these costs
first-class budget items instead of accidents.

## 2. The U-Bus substrate (src/wubu_ubus.c — implemented)

Three parts, each real and tested:

### 2.1 The pool (the unified address space)
`ubus_pool_t` owns flat regions: CART (weights), RDRAM (KV + activations),
OPT (optimizer state). Every engine reads/writes the pool through accessors;
nothing reaches for a raw device pointer except the backend drivers.

### 2.2 The bus meter (measure, don't guess)
`ubus_measure()` runs a microbenchmark per backend at init and fills the
capability table: {gfops, mem_bw, xfer_bw, resident_bytes}. The selector
decisions are made from THESE numbers, so the same binary adapts to a
CPU-only box, a 6GB laptop GPU, or a future 80GB accelerator.

### 2.3 The op dispatch (the agnostic core)
Every op has a signature against the pool and a set of registered backends:
CPU-scalar, CPU-AVX512 (via -march=native), CPU-OpenMP (12 threads),
GPU-cuBLAS (the existing weak-symbol dispatch becomes the GPU backend).
The selector is the roofline model:
    t_backend = max(flops / gfops_b, bytes / bw_b) + overhead_b
Pick the min. The GPU also pays the xfer cost unless the weight cache hits.

Example decisions the selector makes (from the measured table):
- 448x448x128 matmul (25.7 MFLOP): GPU 0.1ms + 2x800KB xfer vs CPU ~1ms
  -> GPU (current threshold ~1 MFLOP).
- tiny 64x64: xfer dominates -> CPU scalar/AVX.
- the head GEMM [vocab x seq x D] = 940 MFLOP: GPU, weight-cached (the
  embedding uploads once per generation, not per call).

## 3. The cartridge (the model as a growable ROM)

The checkpoint is a cartridge: read-mostly, streamed, SWAPPABLE. Growth is
cartridge expansion, not a rewrite:

- **progressive training** (the research: zero/one-layer depth expansion,
  Bu 2025, ~5x compute savings): start tiny, expand depth late (tau ~ 0.8T).
- **function-preserving insertion**: this architecture's gated residual makes
  it natural — a new layer starts with the gate closed (o ~ 0), norms = 1,
  projections near-zero, so the map is the identity at insertion: the loss
  does NOT jump (the DA-verifiable growth invariant). Optimizer state for
  the new layer starts fresh; the WSD schedule absorbs the discontinuity.
- **the growth operator** (the amoeba decision): the trainer emits per-group
  grad norms + the loss EMA slope; the operator grows depth when the slope
  flattens below a threshold AND the batch-relative grad norm of the deepest
  layer exceeds the mean (the layer is "still hungry"). Plateau -> mutate ->
  validate (loss continuity at the insertion point) -> keep or roll back.
- **the cartridge is never "final"**: archive the best, keep the lineage
  (the prestige ledger pattern), RLHF via the oracle loop (the standing
  directive).

## 4. The AGI loop on the U-Bus

corpus -> train -> diagnose -> mutate -> validate -> archive -> RLHF oracle -> repeat
       |        |         |          |         |          |        |
   RDRAM  cart+RDRAM   amoeba    growth     DA-3     ledger    oracles
   streaming  on the bus  diagnostics  operator  FD tests  (wubu-keys)

Every stage is a first-class bus consumer; the diagnosis feeds the mutation;
the mutation is DA-validated before it is kept. The loop is the deliverable,
the U-Bus is the substrate it runs on.

## 5. What "agnostic" means here (the honest scope)

- Backends are REGISTERED capabilities; the selector is a pure roofline
  function over the measured table. Adding AMD/Intel/Apple/NPU = registering
  a backend, not rewriting the engine.
- The model format (the cartridge) is backend-free: floats + a manifest.
- The tests prove EQUALITY across backends (GPU vs CPU within tolerance)
  so a new backend is verified by the same oracle.
- What it does NOT mean: it does not hide the physical reality (VRAM is 6GB,
  PCIe is 8-16GB/s); the selector is honest about xfer costs.

## 6. Implementation status (this session)

- [x] measured roofline table (above)
- [x] src/wubu_ubus.c: pool + meter + roofline selector + CPU/GPU backends
- [x] tools/test_ubus.c: dispatch equality, selector sanity, meter prints
- [x] the wubu training mm rides the U-Bus selector (the weak-symbol GPU
      dispatch = the GPU backend registration)
- [x] the growth operator: wubu_grow_depth (function-preserving) + the
      plateau diagnostics + test_grow (loss continuity at insertion)
- [ ] fp16 tensor path (the 194 TFLOP/s) -- next
- [ ] the GPU attention kernels (the seq-2048 enabler) -- next
- [ ] Gram-NS (Tri Dao 2026) in the NS5 -- next

## 7. References

- the roofline model: data movement >> compute (HPC consensus, the
  c-systems-programming skill's bandwidth diagnostics)
- Gram Newton-Schulz (Zhang/Amsel/Chen/Dao 2026): the NS5 = 2-17% of wall
  time; iterate on the square Gram, up to 2x optimizer speedup
- Deep Progressive Training (Bu, Meta FAIR 2025): zero/one-layer expansion,
  ~5x compute savings, WSD schedule, mixing-time transfer
- Incrementally growing networks (Yuan 2023): dynamic weight/activation/
  gradient stabilization at growth points
- the standing AGI loop (corpus -> train -> diagnose -> mutate -> validate
  -> archive -> RLHF): the amoeba + the DGM pattern

## 8. The console-underground research wave (Kevin-Bacon 7-hop, 2026-08-03)

The user's directive: learn from the hardware-starved developers -- Kaze
Emanuar (N64), the GBA, the PS1, the Dreamcast homebrew (the GTA: Vice
City port), the demoscene -- and fold it into the U-Bus. The hops:

| hop | source | the transferable principle | U-Bus landing |
|---|---|---|---|
| 1 | N64 RDRAM (copetti.org): 4.5MB, 9-bit bus, 640ns latency, 500MB/s streaming | the bus is a STREAM engine: latency is the enemy, never random-access | the pool ops stream; the selector pays xfer costs honestly |
| 2 | Kaze Emanuar: the sine-table MIS-optimization | compute-vs-fetch is a roofline DECISION: when the bus is hot, recompute beats table reads (the table can leave the cache; the polynomial stays in the instruction cache) | the U-Bus selector + `wubu_foldmath` (the compute arm) |
| 3 | Kaze + Silas Lock: the Folded Polynomial | use EVERY symmetry: define on [0,pi/4], fold the circle, ONE sqrt gives the missing value | `include/wubu_foldmath.h`: 8-fold, branchless, no table, no libm; the RoPE tables use it (deterministic, GPU-portable) |
| 4 | GBA (TonC/coranac): 32KB IWRAM 1-cycle vs 256KB EWRAM 6x slower | the memory hierarchy is the program: hot data in the fast tier | the pool tiers (CART/RDRAM/OPT); the residency rules |
| 5 | PS1: 2MB RAM, GTE fixed-point, affine texturing | know the engine's limits and ADAPT THE DATA (subdivide geometry to hide the warp) | the kernel capability profiles; the selector picks per-op |
| 6 | Dreamcast PowerVR TBDR: 32x32 on-chip tiles, no z-buffer in VRAM | tiling = the working set stays on-chip; external memory only at tile boundaries | the flash-attention tiling already in the engine; the principle generalizes |
| 7 | GTA:VC Dreamcast port + the demoscene | a PS2-era game on 8MB: streaming, memory reuse, procedural content; pack bits, trade memory for speed deliberately | the cartridge swap + the pool reuse; the archive discipline |

The convergence: **every generation of starved developers converged on
the same three rules -- stream, tile, fold.** The U-Bus is the substrate
that makes all three first-class: stream (the bus meter + xfer costs),
tile (the on-chip working set), fold (compute-vs-fetch + the folded
math). The foldmath measured truth (this box): the fold is accurate to
~1.5e-7 on the RoPE range and beats PLAIN libm on older stacks; libm
here rides the __svml vector floor (IFUNC, ~4-5 cycles/pair) so the
pure-trig microbenchmark favors libm -- the fold's real value is the
bare-metal WuBuOS target (no libm), the deterministic table, and the
GPU kernels (no libm on-device). The principle, not the microbenchmark,
is the takeaway: "ask what more assumptions you can make" (Kaze).
