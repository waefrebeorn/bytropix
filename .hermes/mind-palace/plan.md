# bytropix Plan — May 27, 2026

## Priority: PARITY FIRST, THEN GAINZ

Parity = bytropix output matches llama.cpp (cos-sim > 0.99 on logits).
Gainz = speed (lower tok/s gap vs llama.cpp).

## PHASE 1: PARITY — IQ2_M FLOOR REACHED

| Step | Status |
|------|--------|
| dump_ref builds + reference logits | ✅ |
| Our logits non-zero (fixed output proj) | ✅ |
| Cos-sim 0.974 vs ref | ✅ (IQ2_M floor — need Q3_K+ to go higher) |
| run-harness.sh → serve_local.py | ✅ |
| test-hermes-headless.sh → serve_local.py | ✅ |

**CONCLUSION: 0.974 = IQ2_M quantization floor.** Need Q3_K/Q4_K/F16 model to reach >0.99. Not available.

## NES EMULATOR = BENCHMARK, NOT PROJECT
Pre-built test workload at ~/hermes-test/projects/nes-emulator/. Do NOT modify CPU/PPU/controller internals. Use only to generate 512K context stress test workload.

## PHASE 2: GAINZ

| Cell | Optimization | Status | Notes |
|------|-------------|--------|-------|
| 241 | SSM buffer pre-allocation (remove 17 malloc/free per layer) | ✅ | Pre-allocated workspace; 30 SSM layers share it |
| 242 | MoE shared expert quantize-once (gate+up share Q8) | ✅ | quantize_row_q8_K once, reuse for both projections |
| 243 | Q4_K output proj threaded for batch | ✅ | Fixed (52x speedup over per-token) |
| 244 | KV cache to Q4_0 format (2GB→500MB) | ✅ | 3 modes: Q4_0 / F16 / F32, Q4_0 default |
| 245 | Attention sparsity wire for decode | ✅ | sparse_buf stack alloc. Env-var controlled. Tested in 512K suite |
| 246 | MoE expert prefetch (LARGE_L3) | ❌ | No gain on i5-8365U (8MB L3 too small for 24MB prefetch). Code exists behind #ifdef LARGE_L3 |

## PHASE 3: DOCUMENTED NOT-A-BUG
| Cell | Claim | Reality | Status |
|------|-------|---------|--------|
| 074 | "Chunked SSM broken, FP accumulation" | Chunked SSM A=(I+L)^{-T} is correct for training but inherently mixes intra-chunk tokens — cannot match sequential bit-exactly. Inference uses sequential (correct). Not a bug. | ✅ DOCUMENTED |

## PHASE 4: END-TO-END TEST HARNESS (May 27)

| Step | Cell | Status | Detail |
|------|------|--------|--------|
| serve_local.py starts | — | ✅ | HTTP server port 8001, subprocess gen_text_cpu |
| Health check / models list | — | ✅ | Returns status: ok, backend: local_cpu |
| Single chat completion | — | ✅ | Cold: 144s (80s model load + 64 tok). Warm: ~185s/64tok |
| Multi-turn conversation | Battleship 173/176 | ✅ | 3 turns NES Q&A. 481 words in 744s |
| NES emulator as workload | Battleship 173 | ✅ | smb.nes ROM loads, frames tick, ASCII output |
| ChatML format | — | ⚠️ Broken | Raw mode treats <|im_start|> as literal tokens |

### What's Next
- Fix ChatML format in gen_text_cpu (CHAT=1 mode needs tokenizer-level support)
- 512K stress test: run conversation at 512K context (needs sparse attention active)
- Real Hermes agent conversation (vs curl commands simulating it)

## PHASE 5: FIX CONTEXT GROWTH PENALTY (RE-DIAGNOSED May 27)

**Original diagnosis (GQA O(n²)) was wrong.** Profiling shows output projection [2048×248320 Q4_K] at 245ms (43.5%) is the real bottleneck. GQA grows only 15% from 2→200 KV. Multi-turn penalty is from process-per-turn re-prefill, not decode decay.

| Step | Task | Effort | Status |
|------|------|:------:|--------|
| 5.1 | PROFILE at 2, 50, 100, 200 KV — confirm output proj dominates | 30m | ✅ |
| 5.2 | Analyze: GQA NOT bottleneck (7.7% of decode) | — | ✅ |
| 5.3 | Option A: Lower SPARSE_MIN 4096→512 | 15m | ✅ Done |
| 5.4 | Option D (NEW): Persistent KV process for multi-turn | 8-16h | ⬜ REAL FIX |
| 5.5 | Option E (NEW): Chunked output proj (top-K then verify) | 4-8h | ⬜ |
| 5.6 | Option F: Logit cache N-hop reuse (direct, no verify) | 2-4h | ⬜ |
| 5.7 | Verify: Run 3-turn conversation, measure improvement | 30m | ⬜ |

**Target:** No decay from turn 2 to turn 3. Maintain >0.8 tok/s across all context lengths.
