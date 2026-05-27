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
| 006 | DDR5 gate: LARGE_L3 compile flag | `#ifdef LARGE_L3` prefetch stride loop for 8 selected experts + shared expert. 256-byte bursts, _MM_HINT_T2. Build: `make gen_text_large_l3` |
| 015 | Router recomputation skip | `moe->precomputed_indices` skips full 2048×256 router matmul. Softmax on 8 selected experts only. ~0.5ms/layer saved. |
| 191 | DA HARD-1 resolved | Router recomputation eliminated via precomputed_indices field. |
| 014 | Router accuracy: normed vs normed2 | Measured ~90% top-8 overlap at 10% noise. Layers 0/20/39 consistent. Architecture validated. |

## Current Bottleneck
- Layer forward: ~200ms (82% of 244ms)
- Output proj (amortized): ~16ms 
- Other overhead: ~28ms

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
