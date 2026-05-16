# WuBuText AI — State Dashboard (May 17 PM v15 — HONEST)

## Ground Truth
**INFERENCE STILL BROKEN BUT 2 NEW BUGS FIXED.** MoE interleaved dequant FIXED, IQ3_XXS block size FIXED, but SSM divergence remains.

Prompt "Hello": our top token "的发展和" (11.68), reference produces coherent text.

## Fixed This Session
- ✅ **IQ3_XXS block size**: `gguf_raw_size` returned 104 bytes/block but actual block_iq3_xxs struct is 98 bytes. Caused MoE down_exps dequant to read 6 bytes past each block → progressively corrupt data. MoE output went from rms=690k to rms=0.25.
- ✅ **IQ4_XS support added**: type 23 was mapped to IQ1_M (wrongly). Fixed enum, added `gguf_raw_size` and `dequantize_iq4_xs_row` function. Affects layers 34, 38, 39.
- ✅ **MoE output verified**: Shared expert rms=0.51 (REASONABLE). Routed expert down weights were garbage (max=4M, mean=14k) due to IQ3_XXS block size. After fix: moe_out rms=0.25, max=2.2 — NORMAL.

## What We Know
- ✅ MoE expert dequant: interleaved [D_MODEL, D_FF, N_EXPERTS] block-by-block extraction FIXED
- ✅ IQ3_XXS block size: 98 bytes (NOT 104)
- ✅ IQ4_XS block size: 136 bytes, dequant function added
- SSM L0 post-MoE residual cos_sim ~0.40 vs reference (unchanged from before)

## Divergence Point
- **SSM output still diverges from reference** at L0 (cos_sim ~0.40). Not caused by MoE since SSM runs before MoE.
- SSM QKV verified matching (cos_sim=1.0). Conv, SiLU, gate match. Recurrence verified by Python script.
- Bug is in the SSM output projection or the post-SSM residual computation.

## Priority
P0 — Find SSM root cause: compare SSM value output (before output projection) vs reference
P1 — Verify cos_sim after L0 post-MoE residual against reference post_moe dump
