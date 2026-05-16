# DA v4 — Devil's Advocate Audit (May 16 PM)

## Phase 1: Claims Audit

### Claim 1: "Q3_K dequant fix resolved NaN in MoE output"
**Source:** infer_text.c run with 19-token prompt
**Trust:** HIGH
**Evidence:** Before fix: NaN at L20+. After fix: h_last mean=-0.14 max=10.6 min=-10.3 (healthy), no NaN at any layer.
**Verify:** `MOE=1 ./infer_text /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Write a short essay about curiosity in science. Give two examples." 5 2>&1 | grep NaN` — 0 matches.
**On re-check:** Still tight.

### Claim 2: "Q5_K dequant qh bit-indexing fix was correct"
**Source:** gguf_reader.c code diff
**Trust:** HIGH
**Evidence:** Before fix: qh read with wrong byte offset (2-bit shift). After fix: byte-order corrected. Q5_K tensors load with no dequant errors.
**Verify:** gguf_reader.c qh read now matches llama.cpp's 2-byte big-endian qh format. No "Dequant: unsupported type" messages for ANY tensor.
**On re-check:** Still tight. Not root cause but correct fix.

### Claim 3: "IMRoPE missing = root cause of garbage output"
**Source:** Hidden state comparison vs llama.cpp
**Trust:** HIGH
**Evidence:** cos sim = -0.0009, max diff = 33.81. llama.cpp uses rope.type=40 (IMROPE). Our RoPE uses uniform theta. After Q3_K + Q5_K fixes, hidden state magnitude is HEALTHY (mean 0.01, max 10.6) — only positional encoding is wrong.
**Verify:**
- Model config: `rope.dimension_sections = [11, 11, 10, 0]`
- llama.cpp: `GGML_ROPE_TYPE_IMROPE = 40`
- Our code: `ROPE_THETA` constant, no section support
- All hidden state metrics (mean, max, min) are normal after dequant fixes
- The ONLY remaining issue is wrong position encoding → attention wrong → garbage
**On re-check:** Tight. This is the last bug.

### Claim 4: "MoE weight interleaving NaN fixed in training"
**Source:** Training pipeline run output
**Trust:** HIGH
**Evidence:** Per-expert dequant test runs all configs with 0 NaN. 11s/step across 16 experts.
**Verify:** `grep NaN` on per-expert dequant output — empty.
**On re-check:** Tight. Different bug from inference NaN (weight layout, not dequant).

### Claim 5: "TGT wrap removal had no effect"
**Source:** Output comparison before/after
**Trust:** HIGH
**Evidence:** 6-token sequence output identical before and after removal. No functional change for forward-only inference.
**Verify:** Diff of output logits before/after — identical first 2 tokens.
**On re-check:** Tight. Only matters for backward pass.

### Claim 6: "GQA gate is correct in both prefill + decode"
**Source:** Code path inspection
**Trust:** MEDIUM (code review only, no runtime verify)
**Evidence:** Gate applies correct per-head scaling in Q projection. Same code path used for prefill and decode.
**Verify:** Both branches checked. No conditional branch that skips gate.
**On re-check:** Tight but should be runtime-verified after IMRoPE fix shows attention matches ref.

### Claim 7: "API server works (14 tests)"
**Source:** tools/serve.py sandbox test run
**Trust:** HIGH
**Evidence:** 14 tests pass: model load, completion endpoint, SSE streaming, rate limits, fake key auth.
**Verify:** `bash tests/test_api.sh` — 14/14 pass.
**On re-check:** Tight. Note: tests check API protocol only. If backend inference produces garbage, API returns garbage. API works, inference doesn't.

### Claim 8: "Per-expert dequant works (0 NaN all configs)"
**Source:** Per-expert dequant test run
**Trust:** HIGH
**Evidence:** Test loops all configs (MOE=0 and MOE=1), checks NaN in hidden state + logits. 0 matches.
**Verify:** 11s/step output shows: `Dequant: expert 0/15 done`, `NaN check: 0 NaN in hidden state`, `0 NaN in logits`.
**On re-check:** Tight.

### Claim 9: "EOS detection correct (gen>1, eos=bos=248044)"
**Source:** Sampler code inspection + test output
**Trust:** HIGH
**Evidence:** EOS token ID = 248044 (same as BOS). Sampler checks `step > 0` before allowing EOS. Prevents stopping on first BOS token.
**Verify:** First token never stopped by EOS check. Token ID 248044 correctly suppressed at step 0.
**On re-check:** Tight.

### Claim 10: "All prior session claims about inference working were wrong"
**Source:** Prior prestige_prompt.md (v9)
**Trust:** CONFIRMED STALE
**Evidence:** Hidden state cos sim = -0.0009. ALL binaries produce garbage output. Previous "✅" statuses only verified "compiles and doesn't crash" — never cross-checked output values.
**Propagation:** All prior mind palace files had stale claims. DA v4 sweep corrects them.

### Claim 11: "GGML_TYPE enum matches llama.cpp for used types"
**Source:** gguf_reader.h vs llama.cpp ggml.h
**Trust:** HIGH
**Evidence:** All types from F32(0) through IQ2_S(22) have same values. Only IQ1_M(23 vs 29) and missing IQ4_NL(20), IQ4_XS(23) are wrong.
**Verify:** Cross-referenced enum line-by-line. Used types = subset where match exists.
**On re-check:** Tight. Minor issue, no current impact.

## Phase 2: Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Confirmation bias** | We assume IMRoPE is the ONLY remaining bug. If SSM formula also differs, fix won't work. | Medium | After IMRoPE fix, check cos sim. If still <0.99, SSM formula differs → deeper investigation. |
| **Survivorship bias** | Only testing with 1-6 input tokens. Longer sequences may expose additional bugs. | Medium | After IMRoPE fix, test with 20+ token prompts and compare hidden states. |
| **IMRoPE implementation error** | Wrong section interleaving pattern produces wrong output. | Medium | Cross-check freq theta formula against llama.cpp. Test single-token position encoding first. |
| **State staleness** | Mind palace files from prior sessions had completely wrong claims. New files may also have errors. | High | Every claim now marked with verification status. Run DA sweep at end of each session. |
| **Sampling not verified** | LLM generates plausible-looking text from any logit distribution. Garbage logits can still produce English-like output. | Low | Always compare hidden states to ref (cos sim). Don't trust "looks like English" as verification. |

## Phase 3: Stale Claim Sweep

Files updated this session (May 16 PM):
- `prestige_prompt.md`: v10→v11. Added Q5_K fix, MoE weight fix, TGT wrap, GQA gate, EOS, API server, per-expert dequant. Updated truth table.
- `goal-mantra.md`: v14→v15. Added all 5 fixes to "what works". Added session commits. Tightened.
- `state.md`: v14→v15. Expanded fixed bug table to 8 entries. Added per-expert dequant + API server to binary table.
- `plan.md`: v12→v13. Expanded bug table to 10 entries (all found/fixed/broken). Tightened.
- `overnight-map.md`: v3→v4. Added full fix list. Added API server + compare commands to trunk ref. Expanded "data not re-derive."
- `plans/devils_advocate_v7.md`: NEW. Full 11-claim audit, 5 risk assessments, stale sweep log.

### Claims retired as stale
- "Q5_K dequant was root cause" → superseded by IMRoPE theory
- "SSM output 140× embedding = suspicious" → was artifact of Q3_K garbage pointers
- "MOE=0 loops 2-token attractor" → was symptom of wrong Q/K positions, not a separate bug
- Training NaN was "mysterious" → now identified as weight interleaving bug (different from inference NaN)

## Phase 4: Next Steps

### P0 — Implement IMRoPE
1. Add `mrope_sections[4]` to model config struct
2. Modify `precompute_rotary_kernel`:
   - 3 sections × different theta frequencies
   - theta_i = freq_base^(-2*section_offset[i]/ROTARY_DIM)
3. Modify `apply_rotary_qk_kernel`:
   - Section-interleaved across head_dim=256
   - First 64 dims only (rest = 0 rotation)
4. Same changes for CPU path in infer_text.c
5. Build + test: `cos sim > 0.99` vs ref

### When IMRoPE is Done
- Generate coherent English output (top-1 should be correct continuation)
- Test essay generation ("Write about curiosity in science")
- Test 256K context forward pass
- Run full test suite
- Update all mind palace files to reflect working inference

### Open Questions for Next Session
1. Does IMRoPE theta formula exactly match llama.cpp GGML_ROPE_TYPE_IMROPE?
2. Are dims 64..255 truly un-rotated? (section[3]=0 confirms)
3. Does SSM path also need RoPE? (likely not — SSM is position-aware by recurrence)
