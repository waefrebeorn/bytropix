# State — May 27, 2026 (Logit Cache + Subset Refresh + LARGE_L3 Gate)

## Branch: cpu-optimize-may26

## Optimization Results
| Optimization | Before | After | Gain | Where |
|-------------|:------:|:-----:|:----:|-------|
| Baseline (no opts) | 2.9 tok/s | — | — | — |
| Logit cache (2:1) | — | 3.5 tok/s | +21% | wubu_model.c |
| + adaptive depth | — | 3.5-4.0 tok/s | +21-38% | wubu_model.c |
| + subset refresh | — | 3.4-4.0 tok/s | +17-38% | wubu_model.c |
| **LARGE_L3 gate** (cell 006) | — | N/A on DDR4 | DDR5/L3>7.4MB target | wubu_model.c + Makefile |

## Cells Completed This Session
| Cell | Vector | Detail |
|------|--------|--------|
| 173 | Benchmark automation | tools/run_benchmark.sh — prefill/decode/PROFILE in one shot. |
| 241 | SSM buffer pre-allocation | Removed 17× malloc/free per SSM layer. `ssm_workspace_t` pre-allocated once in `wubu_model_forward_from_embd`, reused across 30 SSM layers. 64-byte aligned. |
| 244 | KV cache Q4_0 verified | Already active via default `#define KV_CACHE_Q4_0 1` in wubu_model.h. 4:1 compression vs F16. |
| 245 | Attention sparsity verified | `USE_SPARSE_ATTN` env var wired, NSA pattern from DeepSeek-V3.2. |
| 205 | SSM heap allocs resolved | Cell 241 eliminates 17×malloc/layer — covers this gap. |
| 010 | PROFILE all layers | Removed `l < 3` guard — prints all 40 layers now |

## Current Bottleneck
- Layer forward: ~200ms (82% of 244ms) — DDR4 bandwidth bound
- Output proj (amortized): ~16ms 
- Other overhead: ~28ms

## Hardware Frontier
DDR4 (25GB/s): layer forward 200ms is the floor. No remaining cells improve this. Next breakthrough needs DDR5 (50GB/s, ~100ms/layer) or CUDA GPU.

## 300-Gap Battleship
| Status | Count | Notes |
|--------|-------|-------|
| ✅ Completed | 30/300 | Implemented or verified already-fixed |
| 🟢 Trivial (code quality, docs, minor) | ~150 | Mostly 🟢, low ROI |
| 🟡🔴 Core gaps | ~60 | Gyration chain rule, vision, chunked SSM, GPU |
| 🚫 Blocked on hardware | ~60 | Need DDR5 (50GB/s) or CUDA GPU |

## Demoscene Tools Built
- **`quantized_matmul_subset`**: Compute only specified output columns
- **`profile_experts`**: Measure expert stability + frequency
- **`prune_gguf`**: Prune non-essential experts (11.5GB→2.1GB)
- **`delta_sparsity`**: Measure hidden state change between tokens  
- **`build_vocab_clusters`**: Cluster output weight columns (needs balanced k-means)

## Remaining Frontier
- Layer forward (200ms) is now the dominant cost. Memory-bound on DDR4 (25GB/s).
- None of the remaining battleship cells (D, E, F, G, H) directly improve layer speed on this hardware.
- Next breakthrough needs DDR5 (50GB/s) or CUDA GPU.
