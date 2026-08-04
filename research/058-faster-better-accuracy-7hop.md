# research/058 — FASTER + BETTER ACCURACY at low bits: the 7-hop

> 2026-08-04. The user's directive: "seven steps to Kevin Bacon — faster,
> better accuracy for these things so that we can be the best of the best.
> Now that we're designing it, we know what we need to research."
> The design (research/057 mixed compression) tells us exactly what to
> research: the 2-bit accuracy problem and the encode-speed problem.

## The two questions the design raises

1. **Accuracy**: Q8_0 (8.5 bpw) holds 0.99995+ cosine; Q4_0 holds 0.9955.
   The Unsloth ladder wants expert gate/up at ~3 bpw and shared at ~2 bpw
   — how do we keep cosine high at 2-3 bits? The answer the frontier
   converged on: CODEBOOK SHAPE + INCOHERENCE + PER-BLOCK SCALES.
2. **Speed**: naive 2-bit encode = grid search (256 codebooks × 8 dims ×
   scale sweep) ≈ 1M flops per 256-element block. For the seed (35M) that
   is minutes; for the KAHUNA (284B) it is days. How do we encode fast?

## The 7 hops

1. **IQ2_XXS — the codebook block (in-tree)**: the dequant +
   `iq2xxs_grid[256]` (each entry an 8-dim codevector of small ints,
   max ~43) + per-32-element scales (4-bit) + sign bits. 66 bytes per 256
   elements = 2.06 bpw. The grid IS the codebook — accuracy = grid shape.
   Our encoder must search it (or approximate it) fast.
2. **QuIP# / incoherent processing** (arXiv 2306.00588, 2405.19836):
   rotate the weight matrix with a Hadamard transform before quantizing —
   incoherent weights quantize far better at low bits (the outlier
   problem vanishes). WE HAVE `wubu_hadamard.c` (the kernel) and the
   KAHUNA's Config-I proves it in production (TQ3_1S = WHT-rotated 3-bit).
   **The single biggest accuracy lever at 2-3 bits: rotate first.**
3. **AQLM** (arXiv 2401.06118): 2-bit accuracy via MULTIPLE additive
   codebooks per block (2-4 codebooks ≈ 8-bit accuracy at 2-bit size).
   Our grid is one codebook; AQLM says the 2-bit path to high cosine is
   more codebooks, not finer scales.
4. **GPTQ/OBQ second-order** (we have `wubu_gptq`): error compensation
   across columns — the calibration-time accuracy layer (offline only).
5. **Unsloth DQ/UD** (the artifact we dissected, research/057): the
   per-tensor sensitivity ladder picks WHERE the 2-bit goes — half the
   accuracy win is placement, not the block format.
6. **Escha W2 + Config-I** (the validations): 2b gate/up at 12.3 GB
   (#1 HermesAgent-20) and 2.88 bpw at 284B — both keep attention +
   embeddings at high bits (the ladder) and crush the saturated expert
   weights. The "keep maximum what we need" doctrine at scale.
7. **Encode SPEED** (the faster half): (a) SCALE-FIRST — pick d from the
   block amax once, then one-pass grid search per 8-group with the sign
   folded (sign_j = sign(v_j), only the magnitude search remains);
   (b) SIMD the grid dots (the AVX2 kernels exist); (c) parallel blocks
   (the thread_pool exists — encode is embarrassingly parallel);
   (d) skip the sf sweep with a per-32 closed-form scale estimate;
   (e) the LUT trick: grid entries are SMALL INTS — dot(v,g) per grid is
   a 256×8 gather, cache-friendly.

## The convergence (what we build)

**Accuracy at 2-3 bits = Hadamard rotation (incoherence) + codebook
search + per-block scales + the ladder's placement.** Speed = scale-first
+ sign-folded magnitude search + SIMD + parallel blocks.

## The implementation wave (this + next)

1. THIS WAVE: the **IQ2_XXS encoder** in the tensor store (scale-first +
   sign-folded grid search, 256-entry codebook from the in-tree grid) —
   the 2-bit slot of the ladder. Wire into `quant_for_role` (expert
   gate/up + shared → IQ2_XXS). Oracle: cosine vs source > 0.9 at ~2.06
   bpw, size vs Q4_0.
2. NEXT: the **Hadamard-rotate-then-encode** path (reuse wubu_hadamard;
   store the rotation, dequant un-rotates) — the accuracy jump.
3. NEXT: **parallel + SIMD encode** (thread_pool + AVX grid dots) — the
   speed jump; makes the KAHUNA-scale conversion viable.
4. THEN: AQLM-style 2-codebook IQ2 blocks if the single grid plateaus.

## Registration

- INDEX AN10 (this doc): the encoder `wired` when test lands.
