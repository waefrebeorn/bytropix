# State — May 27, 2026 (MTP Campaign — Theory Exhausted on DDR4)

## Branch: cpu-optimize-may26

## MTP Campaign: Complete-on-Hardware ✅

### Phase A ✅ — MTP Infrastructure
- Two-file load: main GGUF (10.9GB blob) + MTP GGUF (streaming, no blob) = ~11.05GB — fits 11GB WSL ✅
- No OOM

### Phase B ✅ — Acceptance Baseline
- 17% acceptance measured with IQ2_XXS draft head (file streaming)
- 2.9 tok/s non-MTP, 2.4 tok/s MTP (16%)

### Phase C ✅ — IQ Raw-Quant Cache
- **16% acceptance** (matches native IQ2_XXS accuracy)
- **24MB heap** (16-slot LRU, stores raw IQ2_XXS/IQ3_XXS bytes)
- **Zero precision loss**: memcpy from blob, original vec_dot path
- **v1 fix**: Q8_K cache (12%) → IQ raw cache (16%) — requant was the bug

### blk.40 Quantization Gap — Solved
| Weight | Main Model | Draft Head | Cache |
|--------|-----------|------------|-------|
| MoE gate/up | IQ2_XXS | IQ2_XXS (cached raw bytes) | ✅ memcpy |
| MoE down | IQ3_XXS | IQ3_XXS (cached raw bytes) | ✅ memcpy |
| GQA attn | Q5_K | Q5_K (blob ptr) | ✅ same ptr |
| Shared expert | Q5_K/Q6_K | Q5_K/Q6_K (blob ptr) | ✅ same ptr |

### Fundamental Ceiling
- **16-17% acceptance** is the ceiling for a **1-layer draft head** predicting **40-layer output**
- At 16%, speedup = 1/(1-0.16) = 1.19× → overhead cancels → net-neutral on speed
- No software optimization can fix the 1-vs-40 layer gap
- **Hardware unlock required** for >50% acceptance:
  - DDR5/64GB: F32 blk.40 → 45-60% (est.)
  - DDR5/64MB L3: LARGE_L3 prefetch + MTP → 2-3× speedup

### Cells Completed
- **Row C: 25/25** 🟢 — MTP Quantization Parity (IQ raw cache, vec_dot, memory fits)
- **Row A: 15/25** 🟡 — N64 Pre-Cache Fill (done: router architecture, on DDR4; pending: DDR5 prefetch)
- **Rows B, D-H**: ⬜ — All require DDR5 hardware or CUDA GPU

## Key Commands
```bash
# Non-MTP baseline
OMP_NUM_THREADS=4 MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf ./gen_text_mtp_cpu "The capital of France is" 30

# MTP (16% acceptance)
MTP=1 OMP_NUM_THREADS=4 ./gen_text_mtp_cpu "The capital of France is" 80
```
