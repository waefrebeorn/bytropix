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

## Working Configuration
- **Build:** `make gen_text_cpu`
- **Run raw:** `MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf ./gen_text_cpu "prompt" 100 40`
- **Run chat:** `CHAT=1 MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf ./gen_text_cpu "prompt" 100 40`
- **Core count:** OMP_NUM_THREADS=4 (4-core i5-8365U)

## Benchmarks (4 cores, 11GB RAM, raw mode)
| Metric | bytropix (before) | bytropix (after) | llama.cpp |
|--------|:-:|:-:|:-:|
| Prefill 4 tok | 1.1 tok/s | **4.3 tok/s** | — |
| Prefill 5 tok | 1.6 tok/s | — | 6.3 tok/s |
| Decode | **2.9 tok/s** | **3.6 tok/s** | 2.7 tok/s |
| Prefill 27 tok (CHAT) | 2.6 tok/s | — | — |
| Output proj (prefill) | 1609ms | **31ms** | — |

Decode beats llama.cpp by 33%. Prefill gap vs llama.cpp (4.3 vs 7.3) reduced from 6x to 1.7x.

## Verified Output
- **RAW:** "The capital of France is Paris." ✓
- **CHAT:** "Here's a thinking process: 1. **Analyze User Input:** ..." ✓

## Remaining CPU Opportunities (ordered)
1. **GQA/SSM projection batching** — Currently projects tokens one-at-a-time in a loop (`for (int s = 0; s < N; s++)`). Use `cblas_sgemm` to batch all N tokens at once for prefill speedup.
2. **Chunked SSM at CS=1** — CS=1 produces exact match vs sequential but no speedup. Only useful for 256K+ context where T is large enough for parallelism to help.
3. **MoE expert prefetch** — API exists but not wired (architecture doc says P2).
4. **Output proj split** — Parallelize Q4_K across threads (architecture doc says P1).

## Files Changed (this session)
- `src/wubu_ssm.c` — tgt_wrap removal + SSM_CHUNK_MIN default 4096
- `.hermes/skills/mlops/cpu-inference-optimization/SKILL.md` — added chunked SSM pitfall

## Context for Next Agent
- Model: `~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf` (11.5 GB, 248K vocab)
- Tokenizer: extracted to `data/vocab.bin` + `data/merges.bin`
- Embeddings: pre-extracted to `data/qwen36_embeddings_c.bin.raw` (2 GB)
- Reference: `~/llama.cpp/build/bin/llama-cli` (BLAS-linked, gives 2.7 tok/s)
- WSL RAM: 11 GB total, model + runtime ~13 GB virtual (swaps)
- WSL config: `/mnt/c/Users/eman5/.wslconfig` (12 GB limit, needs shutdown to apply)
