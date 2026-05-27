# State — May 26, 2026 (MTP Two-File Load + 17% Acceptance Baseline)

## BRANCH: cpu-optimize-may26
**Active fork for CPU optimization work. DO NOT MERGE TO MASTER without user review.**

## MTP Campaign Progress

### Phase A ✅ — MTP Infrastructure Working
- **Problem:** MTP GGUF (11.3GB) + main GGUF (10.9GB) = OOM on 11GB WSL
- **Fix:** Main model loads with full blob (10.9GB). MTP GGUF opened as **secondary ctx without blob** — blk.40 + nextn tensors streamed from file via `gguf_read_tensor_f32` and `gguf_read_raw_tensor`
- `wubu_mtp_free` updated: frees heap-copied quantized weights (attn_q/k/v/output + MoE) when `load_from_blob == false`
- Memory footprint: ~10.9GB (main blob) + ~150MB (MTP head: eh_proj F32 + norms + KV cache) = **~11.05GB — fits** ✓

### Phase B ✅ — Acceptance Baseline Measured
| Metric | Value |
|--------|-------|
| Draft tokens per attempt | 2 |
| Acceptance (fixed EMA α=0.1) | **17%** (8/48 attempts) |
| Acceptance (adaptive α=0.2→0.05) | 12% (worse) |
| Acceptance (α=0.5 aggressive) | 0% (oscillates) |
| Decode speed (no MTP) | 2.3 tok/s (4T, i5-8365U) |
| Decode speed (MTP, 17%) | 2.3 tok/s (net even) |
| Overhead | ~40-50ms per draft call (streaming blk.40 from file) |

**DA: MTP at 17% acceptance is net-neutral at best.** The file I/O overhead of streaming blk.40 MoE (Q2_K/Q3_K, ~100MB per read from SSD each draft call) consumes any gain from speculation.

### Phase C 🔲 — Q8_0 Lazy Dequant Cache
- **Next step**: Cache Q8_0-dequantized versions of blk.40's 8 selected experts in a 12-slot LRU (~40MB RAM)
- **Target**: 25-35% acceptance (from vault paper estimate)
- **Blocking**: Needs implementation of mtp_q8_cache.h with dequant→requant to Q8_0
- Will also eliminate file I/O overhead since Q8_0 weights stay in RAM

### blk.40 Quantization Gap
| Weight | Main Model (layers 0-39) | blk.40 (draft head) |
|--------|--------------------------|---------------------|
| ffn_gate_exps | IQ2_XXS (type 16) | **Q2_K (type 10)** |
| ffn_up_exps | IQ2_XXS (type 16) | **Q2_K (type 10)** |
| ffn_down_exps | IQ3_XXS (type 18) | **Q3_K (type 11)** |
| ffn_gate_inp | F32 (type 0) | **BF16 (type 30)** |
| nextn.eh_proj | N/A | **Q8_0 (type 8)** |

## Remaining CPU Opportunities (ordered)
1. **Output proj split** — 92ms/token decode. At DDR4 bandwidth limit. MTP is next priority.
2. **MTP Q8_0 lazy dequant cache** — Target 25-35% acceptance to break even on throughput.
3. **DDR5 target hardware** — Enables LARGE_L3 prefetch + MTP at tolerable overhead.

## Key Commands
```bash
# Non-MTP baseline
OMP_NUM_THREADS=4 MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf ./gen_text_mtp_cpu "The capital of France is" 30

# MTP (17% acceptance)
MTP=1 OMP_NUM_THREADS=4 ./gen_text_mtp_cpu "The capital of France is" 80
```
