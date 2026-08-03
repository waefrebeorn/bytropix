# 042 — Kaze Emanuar: deep dive + the 7-hop Kevin-Bacon from his work

Status: written 2026-08-03. Sources: Kaze's published videos (transcripts
archived in the wubuos compendium `05-sources/`), the n64decomp scene,
the console-architecture analyses. Purpose: every optimization principle
we extracted lands in the U-Bus design (`research/AGI_LLM_AGNOSTIC_ARCH.md`)
with its human credit attached -- the research is written down, credited,
and hop-traced, not kept in context.

## 1. The individual: Kaze Emanuar (channel @KazeN64)

Kaze Emanuar is a Super Mario 64 ROM hacker and N64 homebrew developer
(~325K subscribers), famous for making impossible things run on a 1995
console. His published body of work:

**The ROM hacks (the motivation for everything):**
- SM64: Last Impact (2016) -- a major full-game ROM hack: new enemies,
  power-ups, stages, music (backloggd catalog).
- Return to Yoshi's Island (2023+) -- a full new campaign with custom
  assets, built on the decomp.
- Many smaller hacks in between; all are his performance test-beds.

**The decomp scene (he is part of the team):**
- The Super Mario 64 decompilation (github.com/n64decomp/sm64) -- a
  two-year community effort (2018-2020) to rebuild the ROM as compilable
  C; motivated by speedrunners wanting to understand the code. He is one
  of the contributors who then USED the source to optimize.

**The optimization videos (his published techniques):**
- "FIXING the ENTIRE SM64 Source Code (INSANE N64 performance)" (2023):
  read every variable, understood every function, edited a 13,000-line
  map file = 100K+ lines read over weeks; renders up to 6x faster than
  vanilla on real hardware; the game logic already ran in <25ms of the
  33.3ms frame -- the win is in the RENDER path and the SHARED MEMORY
  traffic (his core insight: the N64's shared RDRAM transfers memory
  between the render unit (RDP), the RCP, and the CPU -- "the secret is
  in the shared N64 memory").
- "Revolutionizing N64 programming! (SM64 Audio Optimization)" (2024):
  the audio was the ONLY file Nintendo compiled with optimizations on
  (they even ran it through a source-to-source C optimizer) -- and he
  still made it 2x faster (2.4-4.9ms -> ~1.2-2.5ms) by attacking the
  DATA MOVEMENT: "most of the audio processing time is actually spent
  moving large chunks of data around" (echoes, stereo, music mixing).
  He says this changed how he programs the N64 forever.
- "Finding the BEST sine function for Nintendo 64": the lookup-table
  sine was a MIS-optimization -- the RAM bus hits to read the table
  cost more than computing the value (the table can leave the cache;
  the polynomial stays in the instruction cache).
- "The Folded Polynomial" (2023): presented Silas Lock's algorithm
  (see below); 90x accuracy increase on the 4th-order cosine at the
  same speed; the takeaway maxim: "ask yourself what more assumptions
  can you make -- and make sure every assumption you can make is being
  used in a relevant way in your code."
- "How Optimizations made Mario 64 SLOWER": the cache-line/CON layout
  fragility -- a one-line change moves the whole RAM layout and
  everything gets slower; the measurement floor problem (the win was
  "30x smaller than what is even measurable").
- "Mario 64 wastes SO MUCH MEMORY": the 4MB RDRAM is used exactly; the
  cartridge holds only 8MB.
- The N64 Programming PRIMER series: "Learn how to write code for
  Nintendo 64!" -- a from-scratch N64 development tutorial series.

**The collaborators and influences (credited):**
- Silas Lock -- the folded polynomial (joined Kaze's Discord, posted the
  idea "with a great mathematical proof for its accuracy"; Kaze says
  only one person thought of it). The numerics (minimax-style
  coefficient fitting) came from community suggestions.
- The n64decomp team (Kenix, Rozlette, the ZRET Zelda spin-off, the
  decomp.me collaborative decomp site).
- The SM64 ROM-hack community that stress-tests every optimization on
  real hardware.

## 2. The folded polynomial (Silas Lock, via Kaze) -- the math, written down

Define ONE even polynomial P(t) ~ cos(t) on the first EIGHTH of the
circle, t in [0, pi/4]. Use EVERY symmetry of the sine wave:
1. even symmetry: P valid on [-pi/4, pi/4].
2. sin(x) = cos(x - pi/2): covers [pi/4, 3pi/4].
3. mirror: covers [3pi/4, pi] and [-pi, -3pi/4].
4. mirrored sine: covers [-3pi/4, -pi/4].
5. the missing value (sin from cos or cos from sin) comes from ONE
   sqrt of (1 - v^2) -- the Pythagorean identity, which also guarantees
   the vector is normalized to exactly one.
Result: a SECOND-ORDER polynomial more accurate than the previous
THIRD-order one (the smaller range lets the coefficients fit better) --
3x more accurate for graphics, 90x for the physics cosine, at 63.5
cycles average for both sin+cos on the N64 (vs 65 before). The quadrant
shift is done with "extremely efficient bit math" (the game's angles are
fixed-point 0..65535 -- the quarter IS a bit shift).

Our implementation: `include/wubu_foldmath.h` (header-only static
inline -- the only way the vectorizer can SIMD it). We kept the Taylor
poly pair (sin odd + cos even) instead of the sqrt because on Zen 4 the
sqrt costs more than the second polynomial; measured ~1.5e-7 accuracy on
the RoPE range; the RoPE tables build with it (deterministic, no libm,
GPU-portable). Honest measured note: glibc's __svml IFUNC floor wins the
pure-trig microbenchmark on this box; the fold's value is the bare-metal
WuBuOS target (no libm), determinism, and the compute-vs-fetch
principle.

## 3. The 7-hop Kevin-Bacon from Kaze's work

| hop | node | what the hop gave us | U-Bus landing |
|---|---|---|---|
| 0 | Kaze Emanuar (seed) | the folded polynomial, the shared-memory secret, the data-movement audio fix, the "assumptions" maxim | the U-Bus + foldmath (built) |
| 1 | the SM64/N64 decomp scene (n64decomp, decomp.me, ZRET, Kenix, Rozlette) | 100K-line code reading, the IDO compiler archaeology, collaborative decomp as a METHOD | the "read everything, then optimize" discipline; the FD tests as the decomp-equivalent oracle |
| 2 | the N64 hardware (RSP microcode, RDP, RDRAM, the 9-bit bus; copetti.org) | the 640ns-latency stream bus, the RSP vector co-processor, the RDP fixed-function engine | the U-Bus pool (cartridge/RDRAM) + the backend registry |
| 3 | the SGI heritage (the IDO compiler, the Reality Co-Processor, the graphics-workstation lineage) | the compiler matters (IDO 7.1 vs 7.2 changed everything); the workstation -> console -> GPU lineage | the compiler-flag discipline; the CFLAGS as a first-class engine concern |
| 4 | the console-adjacent underground (GBA TonC/coranac, PS1 GTE, Dreamcast PowerVR, the GTA:VC port) | IWRAM/EWRAM hierarchy, affine-subdivide adaptation, tile-based deferred rendering, the "make it fit" port | the pool tiers, the kernel capability profiles, the tiling rule |
| 5 | the demoscene (64k intros, Assembly) | procedural content beats storage; trade memory for speed deliberately | the cartridge as PROCEDURE (the growth operator generates weights) |
| 6 | the PC engine lineage (id/John Carmack's fast inverse sqrt) | the SAME fold principle in the exponent domain: 0x5f3759df = the bit-cast fold of the range | the foldmath as a FAMILY (domain fold, exponent fold, precision fold) |
| 7 | the modern roofline + GPU kernel optimization (SVML floor, CUDA graphs, flash-attention tiling) | the SVML/sincos8 floor, the command-list pattern, tile-based attention | the U-Bus selector (measured table), the op-queue, the tiling |

## 4. The convergence (what the 7-hop agrees on)

Every constrained-system community independently converged on the same
law: **when resources are fixed, fold the domain until the problem
fits.** Kaze/Silas folded the ANGLE domain; Carmack folded the EXPONENT
domain; the demoscene folds the STORAGE domain (procedural generation);
the DSPs fold the PRECISION domain (fixed-point); the N64 devs fold the
MEMORY domain (shared RDRAM, stream-only). The U-Bus's three rules --
stream, tile, fold -- are instances of this one law:
- stream = fold the ACCESS pattern (never random-access a stream bus),
- tile = fold the WORKING SET (on-chip, tile-local),
- fold = fold the DOMAIN (compute-vs-fetch, symmetry, precision).

## 5. Credits (every individual whose work is in the U-Bus)

- Silas Lock -- the folded polynomial algorithm (2023, via Kaze's Discord).
- Kaze Emanuar -- the N64 optimization corpus, the sine-table
  mis-optimization proof, the shared-memory secret, the data-movement
  audio fix, the "assumptions" maxim; the SM64 decomp contributions.
- Rodrigo Copetti -- the console architecture analyses (N64, GBA, PS1,
  Dreamcast) that grounded the hardware facts.
- The n64decomp team (Kenix, Rozlette, et al.) + decomp.me -- the
  decomp-as-method.
- Jack Zhang, Noah Amsel, Berlin Chen, Tri Dao -- Gram Newton-Schulz
  (2026): the square-space NS iteration (next optimizer upgrade).
- Zhiqi Bu (Meta FAIR) -- Deep Progressive Training (2025): zero/one-
  layer depth expansion, ~5x compute, WSD (the growth operator).
- X. Yuan et al. (NeurIPS 2023) -- incrementally growing networks:
  dynamic weight/activation/gradient stabilization at growth points.
- TonC/coranac (GBA) -- the IWRAM/EWRAM hierarchy discipline.
- The GTA: Vice City Dreamcast port team -- the 8MB "make it fit" port
  (vector-math optimizations, streaming, memory reuse).
- The demoscene -- procedural content, deliberate memory trade-offs.

The engine-side landing: `research/AGI_LLM_AGNOSTIC_ARCH.md`,
`include/wubu_foldmath.h`, `src/wubu_ubus.c`. The archive: wubuos
compendium `05-sources/kaze-emanuar.md` + the raw transcripts.
