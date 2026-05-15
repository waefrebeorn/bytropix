# WuBuText AI — Overnight Navigation Map (May 15 PM v6 — COMPREHENSIVE)

## Where We Are

**P0-P2 all complete.** Per-block IQ2_XXS extraction: full dequant eliminated, 177s→11s/step. Multi-flag verification: all 6 env flags tested individually + combined, 0 NaN. MoE output magnitude resolved (hidden max=13, old 5e9 from buggy strided extraction). Memory optimization: persistent buffers in lmoe_t, no per-step 3GB alloc/free.

**Tailslayer (May 15):** Hedged-read C++ library analyzed. Direct pattern match for speculative decode kernel. See `vault/tailslayer/` and `plan.md`.

**Paper audit (May 15):** 32 Qwen papers cross-referenced vs C headers. 9/14 config params match. 2 ❌ missing (MRoPE, MTP). 1 ❌ discrepancy (CONV_DIM). 2 Verify (RoPE theta, bos/eos). See `state.md`.

### Verified Claims (May 15 session)

- ✅ gguf_raw_size(IQ2_XXS) fixed: 72→66 bytes/block (verified empirically)
- ✅ Per-expert dequant: 3.9ms/expert, bit-exact match with full dequant reference
- ✅ train_integrated avg: 11s/step (was 177s), CE 21.6→18.4, 0 NaN
- ✅ All 6 flag combos verified: TST/RSGD/PGA/NSSM/NMOE/POINCARE_R
- ✅ Hidden magnitudes: max=13, rms=2.6, dim2035~2-5 (resolved)
- ✅ Persistent MoE cache: 3GB calloc/free eliminated from step loop
- ✅ Tailslayer: 5 files analyzed, pattern map complete
- ✅ Paper audit: 14 discrepancies catalogued

### Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| ~11s/step GPU compute (40 layers) | Performance | GPU forward, not dequant bottleneck |
| PGA loss jumps 21.6→69 | Numeric | Pre-existing LR issue |
| CPU output projection ~2s/token | Performance | O(N·V·D) for V=248320 |
| CONV_DIM=8192 vs config 1536 | Possible bug | Needs code audit |
| MRoPE 3D not implemented | Correctness | P2 |
| MTP head missing | Feature | P3 |

### Build Command
```bash
cd /home/wubu/bytropix && make train_integrated
```

### Fallback
If blocked on perf: GPU MoE forward (upload expert weights → CUDA matmul)
If blocked on PGA: reduce lr_gqa from lr*0.01 to lr*0.001
If blocked on CONV_DIM: read wubu_ssm.h, compare with config.json

### Priority for Next Session
1. P0: GPU MoE forward (biggest perf win)
2. P1: PGA LR tuning (fix loss jump)
3. P2: Tailslayer spec-decode kernel, MRoPE, sparse attention port
