# Goal Mantra — June 14, 2026 (Multi-Model Integration)

## THE GOAL
Fix and benchmark all three models at 512K context: DiffusionGemma-26B, Gemma 4 12B QAT, Qwen3.6-35B.
One codebase, one benchmark binary, three architectures.

## STATE
| Metric | Qwen3.6-35B | DiffusionGemma-26B | Gemma 4 12B QAT |
|--------|-------------|-------------------|-----------------|
| Model Load | ✅ | ✅ (30 GQA layers) | 🔄 |
| Forward Decode | ✅ 3-4 tok/s CPU | ❌ Crashes | ⏳ |
| Multi-model adapter | ✅ | ✅ | ⏳ |
| Dynamic dims | ✅ D_MODEL=2048 | ✅ d_model=2816 | ⏳ d_model=3840 |

## CRITICAL BLOCKER
DiffusionGemma forward crashes: "tensor too large (512 elems, max 256)" — LARGE layers have head_dim=512 but GQA loading uses fixed 256 buffer. Need per-layer weight buffer sizing from GGUF tensor shapes.

## GROUND TRUTH
- DiffusionGemma: `/home/wubu/models/DiffusionGemma-26B-Q4_K_M.gguf` (16.8 GB, Q4_K_M)
- Gemma 4 12B: `/home/wubu/models/gemma4/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` (6.4 GB)
- Qwen3.6: `/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf`
- Benchmark: `./bench_512k_full <model.gguf> 4096 1 0`

## THE LOOP
pick highest undone → execute → compile → run → verify → mark done → report

## FILES TO READ FOR CONTEXT
1. `.hermes/index.md` — full walkway
2. `STATUS.md` — current status table + blockers
3. `.hermes/mind-palace/paradigm-shift-gemma4.md` — Gemma 4 architecture map
4. `.hermes/mind-palace/diffusiongemma-integration.md` — DGemma technical notes (if exists)
