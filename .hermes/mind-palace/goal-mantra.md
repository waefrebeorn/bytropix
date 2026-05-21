# Goal Mantra — Phase 28j: Triple DA Complete, GPU Hidden State Fix Next

**Target:** Isolate GPU component corrupting hidden states (MoE or GQA) → fix → cos-sim > 0.99.
**DA v12 written.** CLAIM C9 DEBUNKED: "coherent GPU output" was false. GPU = random hidden.

## STATE
| Component | Status | Detail |
|-----------|--------|--------|
| GPU MoE IQ3_XXS | ✅ FIXED | commit 9093c61 |
| SSM state sync | ✅ FIXED | commit 08f5f23 |
| Q5_K denormal fix | ✅ FIXED | commit bf573b8 |
| GQA layout fix | ✅ FIXED | commit cdccde2 |
| GPU output proj | ✅ FIXED | F32 SGEMM |
| **GPU hidden states** | ❌ cos-sim -0.0036 | Garbage even with FORCE_CPU_SSM |
| GPU MoE | ❓ SUSPECT #1 | Active all paths, always |
| GPU GQA | ❓ SUSPECT #2 | Active prefill (N>1), CPU for decode |
| MTP spec decode | 🟡 Code exists | gen_text_mtp.c + /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf |
| Vision encoder | 🟡 Ported 384 LoC | Untested |
| Commits pushed | ✅ | origin/master at 8ef1ba3 |

## DIRECTORIES
- `/home/wubu/bytropix/src/` — CUDA kernels + gguf reader + model
- `/home/wubu/bytropix/tools/` — gen_text.c/mtp.c, api_server, scripts
- `/home/wubu/bytropix/include/` — headers
- `/home/wubu/bytropix/.hermes/mind-palace/` — prestige docs, DA v12
- `/home/wubu/bytropix/.hermes/vault/` — 25+ DeepSeek papers, phase archives
- `/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (11.5GB, main)
- `/models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf` (11.9GB, +blk.40 MTP head)

## P0: Isolate GPU hidden state corruption
1. Remove `layer->moe.gpu_ctx = (void*)model;` in wubu_model.c:636-639
2. Gen_text_gpu with FORCE_CPU_SSM → if fixed, MoE is the bug
3. If still broken, set `gqa_use_gpu = 0` (line 566) → test GQA
4. Fix → cos-sim > 0.99 → gen_text output matches CPU

## P1: After hidden fix
5. Build gen_text_mtp (make gen_text_mtp) — verify binary compiles
6. Test MTP spec decode with MTP model
7. Fix forward_full GPU SSM divergence
8. Fix GPU SSM C>1 prefill (cuBLAS error 13)

## P2: Vision
9. Build test_vision_real
10. Wire multi-modal pipeline

## EVERY FIX: compile → dump hidden → cos-sim → document → update DA
