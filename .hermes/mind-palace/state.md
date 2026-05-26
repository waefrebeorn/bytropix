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

## Benchmarks (4 cores, 11GB RAM, raw mode)
| Metric | bytropix (fixed) | llama.cpp |
|--------|:-:|:-:|
| Prefill 5 tok | 1.5 tok/s | 6.3 tok/s |
| Prefill 6 tok | 1.3 tok/s | — |
| Decode | 2.6-2.8 tok/s | 2.7 tok/s |

Decode slightly below llama.cpp parity. Prefill gap needs MoE expert prefetch or output proj split.

## Verified Output
- **RAW:** "The capital of France is Paris." ✓
- **RAW:** "Write a poem about AI" → thinking process output ✓
- **CHAT:** TBD

## Remaining CPU Opportunities (ordered)
1. **MoE expert prefetch** — API exists but not wired (arch doc P2). Currently OMP threads wait for next expert in sequential unlock. Should fetch next expert's weight pointer during current expert's F32 down-projection.
2. **Output proj split** — Parallelize Q4_K across threads (arch doc P1).
3. **Chunked SSM at CS=1** — Only useful for 256K+ context. No speedup at short lengths.

## Files Changed (this session)
- `src/wubu_ssm.c` — Removed buggy AVX2 conv1d; batched GQA projections (Q+gate, K, V)
- `.hermes/mind-palace/state.md` — Updated with conv1d bug fix + GQA batching

## Context for Next Agent
- Model: `~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf` (11.5 GB, 248K vocab)
- Tokenizer: extracted to `data/vocab.bin` + `data/merges.bin`
- Embeddings: pre-extracted to `data/qwen36_embeddings_c.bin.raw` (2 GB)
- Reference: `~/llama.cpp/build/bin/llama-cli` (BLAS-linked, gives 2.7 tok/s)
- WSL RAM: 11 GB total, model + runtime ~13 GB virtual (swaps)
- WSL config: `/mnt/c/Users/eman5/.wslconfig` (12 GB limit, needs shutdown to apply)