# state — May 16 v3 DA deep-dive session

## Done this session
- Full component-by-component deep dive: SSM, GQA, MoE, RoPE, dequant, weight indexing
- Verified infer_text.c inference pipeline structure correct
- Identified wubu_gqa_forward() weight indexing bug (dead code for inference)
- Verified SSM recurrence matches llama.cpp delta-net-base structure
- Verified MoE router/shared expert structure
- Verified RoPE applied in both prefill + decode paths
- Identified 5 remaining active suspects with severity ranking

## DA AUDIT — Survivorship Bias Strip
All P0-P3 items were "done" at "compiles" level only. None verified against llama.cpp output.

### Phase 0: Correctness Fixes
❌ Shared expert gate — compiled only. Never verified cos-sim against llama.
❌ MoE contiguous dequant fix — wrote coherent English output once but still wrong.
❌ MOE=1 default — trivial env var change, uncontroversial.
❌ MAX_LAYERS=0 clamp — trivial, uncontroversial.

### Phase 1: Speed
❌ P1a Chunked DeltaNet — training-only, never verified against sequential path.
❌ P1b Fused Gate+Up — separate weights, design decision, no verification needed.
❌ P1c Single-Pass Top-K — compiled only. Same result as bubble sort (same algorithm).

### Phase 2: GPU
❌ P2a Warp CUDA scan — compiled only. Not used in current CPU-only test path.
❌ P2b Conv state kernels — compiled only.
❌ P2c Conv1d shared mem — compiled only.
❌ TF32/block 512 — speed only, doesn't affect correctness.

### Phase 3: Quant
❓ P3a IQ2 on-the-fly dot — 4/4 unit tests pass. Scale: tiny test, not full-model verified.
❓ P3b K-Quant support — raw_size + dequant functions exist. Never verified against llama.cpp.

### Auto-embedding + BOS
❌ Auto-embedding — extracted token_embd from GGUF at load time. Verified file sizes match. But embedding correctness relies on dequant correctness (type 0 = F32).
✓ BOS: ADD_BOS env var default off matches add_bos_token=false — confirmed.

### Critical: Output Still Wrong
❌ "Hello" → "Plot" — root cause unfixed after 2 sessions of fixes.
❌ Root cause narrowed to 5 suspects with likelihood ranking.

## 5 Active Suspects (prioritized)
1. [🔴 HIGH] Q5_K dequant — 181 tensors. One bad block × millions = wrong output.
2. [🔴 HIGH] Output weight (type 12 Q4_K) dequant — one bad block corrupts all logits.
3. [🟡 MED] SSM Q scaling 1/sqrt(128) — verify llama.cpp applies same factor.
4. [🟡 MED] RMSNorm epsilon — bytropix 1e-6 vs llama.cpp.
5. [🔵 LOW] TGT wrapping in GQA attention — clips scores to [-π,π]. Not in llama.cpp.

## Clean Findings
- Weight indexing in infer_text.c SSM + GQA: CORRECT (i + j*D_MODEL pattern)
- RoPE in infer_text.c: CORRECT (both prefill + decode)
- MoE router: CORRECT structure
- SSM recurrence: CORRECT structure vs delta-net-base.cpp
- wubu_gqa_forward() weight indexing: BROKEN but dead code for inference
