# bytropix Mind Palace — May 26, 2026

## Active Branch
- `cpu-optimize-may26` — CPU optimizations (batching, AVX2 norms/conv1d, OMP fix)
- Master push-guarded (remote URL = `no-push`)

## Core Documents
| Document | Purpose | Status |
|----------|---------|--------|
| `state.md` | Current performance state, benchmarks, known bugs | ✅ |
| `plan.md` | Next steps, priority | ✅ |
| `prestige_prompt.md` | Session resume prompt | ✅ |
| `goal-mantra.md` | Goal and operating loop | ✅ |
| `bytropix-accomplishments-vaulted.md` | All completed work archived | ✅ NEW |
| `bytropix-300-gap-battleship.md` | 300-gap fresh analysis | ✅ NEW |

## Optimization Status
- **Prefill:** 1.1 → 4.3 tok/s (3.9x) — batched projections + AVX2 norms
- **Decode:** 2.5 → 3.6 tok/s (44% gain) — outer loop in MoE freed cores
- **Output proj:** 1609ms → 31ms (52x) — nested OMP fix
- **DRAM refresh:** 7.62µs, 2.4% stalls — negligible impact

## Key Paths
- Models: `~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf` (10.7 GB)
- Embeddings: `~/bytropix/data/qwen36_embeddings_c.bin.raw` (2 GB)
- Tokenizer: `~/bytropix/data/vocab.bin` + `~/bytropix/data/merges.bin`
- Reference: `~/llama.cpp/build/bin/llama-cli` (BLAS-linked)
- Tailslayer: `~/tailslayer/` (DRAM channel-hedged reads, research only)

## Build
```bash
cd ~/bytropix && make gen_text_cpu
OMP_NUM_THREADS=4 MODEL=~/models/... ./gen_text_cpu "prompt" 100 40
```