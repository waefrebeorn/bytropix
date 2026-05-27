# State — May 26, 2026 (CPU Inference Accuracy Fixed)

## BRANCH: cpu-optimize-may26
**Active fork for CPU optimization work. DO NOT MERGE TO MASTER without user review.**

## Bugs Fixed This Session

### 🔴 P0: Chunked SSM FP Accumulation (FIXED)
- **Root cause:** Commit `501518f` wired chunked SSM recurrence into forward pass with `SSM_CHUNK_MIN=2` (CS=2). FP accumulation errors amplify through 30 SSM layers causing wrong token selection.
- **Fix:** Changed default SSM_CHUNK_MIN from 2 → 4096 (wubu_ssm.c:504). Sequential path always used for short prefills.
- **Symptoms before fix:** RAW mode produced byte-level garbage ("----"). CHAT mode degenerated after 5-10 tokens ("other... is is was was...").
- **Env var workaround:** `FORCE_CPU_SSM_SEQ=1` or `SSM_CHUNK_MIN=999999`

### 🔴 P0: tgt_wrap on Attention Scores (FIXED)
- **Root cause:** `tgt_wrap(x) = fmod(x+π, 2π)-π` applied to Q·K dot products before softmax (wubu_ssm.c:1476,1664). Wraps large positive scores (best matches) to negative values → inverts attention weights.
- **Fix:** Removed both `tgt_wrap` calls. Normal Q·K * scale scores passed to softmax.
- **Symptoms:** Best match → 2% probability, worst match → 76% probability.

### 🔴 P0: AVX2 Conv1d Kernel Broadcast Bug (FIXED May 26 Session 2)
- **Root cause:** Commit `9d4029c` added AVX2 conv1d using `_mm256_set1_ps(kernel[ki + c * k])` which broadcasts channel-c's kernel value to all 8 vector lanes. Each channel has its OWN kernel values — broadcasting produces wrong output for channels c+1 through c+7.
- **Fix:** Removed AVX2 conv1d path. Sequential per-channel conv1d always used.
- **Symptoms:** Same as pre-tgt_wrap-fix: byte-level garbage output ("infi doss！\"..."). Confused with prior bugs — only found by bisecting 066ff74 (working) vs 9d4029c (broken).
- **Also affecting:** GQA forward projection was per-token; now batched via `quantized_matmul_batched` (same pattern as SSM forward).

## Optimizations Applied This Session
1. **GQA projection batching** — GQA Q+gate, K, V projections now use `quantized_matmul_batched` (was per-token quantize+matmul). Same batching pattern as SSM forward. Applied to both `wubu_gqa_forward` and `wubu_gqa_forward_save`.

## Working Configuration
- **Build:** `make gen_text_cpu`
- **Run raw:** `MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf ./gen_text_cpu "prompt" 100 40`
- **Run chat:** `CHAT=1 MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf ./gen_text_cpu "prompt" 100 40`
- **Core count:** OMP_NUM_THREADS=4 (4-core i5-8365U)

## Optimizations Applied This Session
1. **GQA projection batching** — GQA Q+gate, K, V projections now use `quantized_matmul_batched` (was per-token quantize+matmul). Same batching pattern as SSM forward. Applied to both `wubu_gqa_forward` and `wubu_gqa_forward_save`.
2. **MoE shared expert Q8 reuse** — Quantize x_s once per token, use `quantized_matmul_from_q8` for both gate+up projections (was 2 separate quantize+matmul calls).
3. **MoE routed expert Q8 reuse** — Same pattern inside each OpenMP task: quantize x_s once, use `quantized_matmul_from_q8` for both gate+up of each routed expert (was 2 quantize calls per expert).

## Benchmarks (4 cores, 11GB RAM, raw mode)
| Metric | bytropix (fixed) | llama.cpp | Gain vs Prior |
|--------|:-:|:-:|:-:|
| Prefill 5 tok | **3.3 tok/s** | ~6.3 tok/s | **2.4×** (was 1.4) |
| Decode | **2.7 tok/s** | ~2.7 tok/s | **1.04×** (was 2.6) |

Decode at DDR4 memory bandwidth wall (~2.3 tok/s theoretical). Prefill gap to llama.cpp reduced to ~2×.

## Verified Output
- **RAW:** "The capital of France is Paris." ✓ (both prefill and decode)
- **CHAT:** TBD

## Remaining CPU Opportunities (ordered)
1. **Output proj split** — 92ms/token (25% of decode). Q4_K quantized matmul is already parallelized across threads. At DDR4 memory bandwidth limit, further gains need data reduction (IQ1_M quant → 15% less data) or MTP spec decode.
2. **Chunked SSM at CS=1** — Only useful for 256K+ context. No speedup at short lengths.
3. **SSM buffer pre-allocation** — 12 malloc/free per layer (negligible, ~36μs/token).

## Files Changed (this session)
- `src/wubu_ssm.c` — Removed buggy AVX2 conv1d; batched GQA projections (Q+gate, K, V)
- `src/wubu_moe.c` — Q8 reuse for shared expert + routed experts (gate+up projections)
- `src/quantized_matmul.c` — (unrelated fixes)
- `.hermes/mind-palace/state.md` — Updated with conv1d bug fix + GQA batching + MoE Q8 reuse

## Context for Next Agent
- Model: `~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf` (11.5 GB, 248K vocab)
- Tokenizer: extracted to `data/vocab.bin` + `data/merges.bin`
- Embeddings: pre-extracted to `data/qwen36_embeddings_c.bin.raw` (2 GB)
- Reference: `~/llama.cpp/build/bin/llama-cli` (BLAS-linked, gives 2.7 tok/s)
- WSL RAM: 11 GB total, model + runtime ~13 GB virtual (swaps)
- WSL config: `/mnt/c/Users/eman5/.wslconfig` (12 GB limit, needs shutdown to apply)