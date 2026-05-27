# State — May 27, 2026 (Q8_0 Cache Phase C ✅ — 12% acceptance)

## BRANCH: cpu-optimize-may26

## MTP Campaign Progress

### Phase A ✅ — MTP Infrastructure Working (two-file load, no OOM)

### Phase B ✅ — Acceptance Baseline Measured (17% baseline)

### Phase C ✅ — Q8_0 Lazy Dequant Cache
- **Status**: Implemented and verified
- **Cache**: 12-slot LRU, ~41MB heap, Q8_K×Q8_K matmul for expert dot products
- **Acceptance**: **12%** (vs 17% baseline with pure IQ2_XXS draft head)
- **Q8_K overhead**: ~3% acceptance loss vs pure IQ2_XXS vec_dot path
- **Crash**: Fixed buffer overflow bug (MTP_Q8_WEIGHT_BYTES used 34/32 ratio → 1.0625, actual block_q8_K is 292/256 = 1.140625)
- **No crash on exit** ✅
- **Baseline (no MTP)**: 2.9 tok/s (4T), MTP: 2.3 tok/s (12% acceptance still below break-even)

### blk.40 Quantization Gap
| Weight | Main Model (layers 0-39) | blk.40 Draft Head | Q8 Cache Used |
|--------|--------------------------|-------------------|---------------|
| ffn_gate_exps | IQ2_XXS | **Q8_K** (cached) | ✅ |
| ffn_up_exps | IQ2_XXS | **Q8_K** (cached) | ✅ |
| ffn_down_exps | IQ3_XXS | **Q8_K** (cached) | ✅ |
| GQA attn q/k/v/output | Q5_K | Q5_K (blob ptr) | N/A |
| Shared expert | Q5_K/Q6_K | Q5_K/Q6_K (blob ptr) | N/A |

### Next Priority
1. **Acceptance too low** (12% vs 45-60% target). Q8_K cache helps with memory but high-precision draft head needed for speedup.
2. **MTP still slower than non-MTP** at 12% acceptance. Need >35% acceptance.
3. **DA suggests**: The fundamental issue is 1-layer draft head can't predict 40-layer output. Consider:
   - Warm-start cache (pre-fill most-common experts)
   - Custom expert selection for blk.40 that mimics main model patterns
   - Full F32 blk.40 (requires DDR5 machine with 64GB)
4. DDR5 target hardware remains the only path to >50% acceptance.

## Key Commands
```bash
# Non-MTP baseline
OMP_NUM_THREADS=4 MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf ./gen_text_mtp_cpu "The capital of France is" 30

# MTP (17% acceptance)
MTP=1 OMP_NUM_THREADS=4 ./gen_text_mtp_cpu "The capital of France is" 80
```
