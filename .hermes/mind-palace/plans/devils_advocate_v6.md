# DA v3 — Devil's Advocate Audit (May 16 AM)

## Phase 1: Claims Audit

### Claim 1: "Q3_K dequant fix resolved NaN in MoE output"
**Source:** infer_text.c run with 19-token prompt
**Trust:** HIGH
**Evidence:** Before fix: NaN at L20+. After fix: h_last mean=-0.14 max=10.6 min=-10.3 (healthy), no NaN at any layer.
**Verify:** `MOE=1 ./infer_text /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Write a short essay about curiosity in science. Give two examples." 5 2>&1 | grep NaN` — 0 matches.

### Claim 2: "Dequant no longer errors (Q3_K supported)"
**Source:** gguf_reader.c code + test run
**Trust:** HIGH
**Evidence:** No "Dequant: unsupported type 16" or "Dequant: unsupported type 11" messages in output.
**Verify:** `grep -i "unsupported type"` on test output — empty.

### Claim 3: "IMRoPE missing = root cause of garbage output"
**Source:** Hidden state comparison vs llama.cpp
**Trust:** HIGH
**Evidence:** cos sim = -0.0009, max diff = 33.81. llama.cpp uses rope.type=40 (IMROPE). Our RoPE uses uniform theta.
**Verify:** 
- Model config: `rope.dimension_sections = [11, 11, 10, 0]`
- llama.cpp: `GGML_ROPE_TYPE_IMROPE = 40`
- Our code: `ROPE_THETA` constant, no section support
- Without correct positional encoding, attention computes wrong Q/K dot products

### Claim 4: "GGML_TYPE enum values match llama.cpp for used types"
**Source:** gguf_reader.h vs llama.cpp ggml.h
**Trust:** HIGH
**Evidence:** All types from F32(0) through IQ2_S(22) have same values. Only IQ1_M(23 vs 29) and missing IQ4_NL(20), IQ4_XS(23) are wrong.
**Verify:** Cross-referenced enum line-by-line against llama.cpp ggml.h.

### Claim 5: "Q5_K dequant fix from previous session was correct"
**Source:** Session log: "qh bit-indexing FIXED"
**Trust:** MEDIUM (was claimed before but not root cause)
**Evidence:** Q5_K tensor output.weight loads fine (no errors). But the Q5_K fix addressed a byte-ordering issue that produced qh values shifted by 2 bits. It changed the output but didn't fix the root cause because the REAL bug was Q3_K dequant missing.
**Verify:** The Q5_K fix is NOT the root cause (model still produced garbage after fix). The Q3_K fix WAS needed but ALSO not sufficient. Now we know IMRoPE is the remaining issue, and Q5_K/Q3_K dequants are both correct.

### Claim 6: "All prior session claims about inference working were wrong"
**Source:** Prior prestige_prompt.md (v9) claimed inference was working
**Trust:** CONFIRMED STALE
**Evidence:** Hidden state cos sim = -0.0009. ALL binaries produce garbage output. Previous "✅" statuses only verified "compiles and doesn't crash" — never cross-checked output values.
**Propagation:** All prior walkway files had stale claims about working inference. This DA sweep corrects them.

## Phase 2: Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Confirmation bias** | We assume IMRoPE is the ONLY remaining bug. If SSM formula also differs, fix won't work. | Medium | After IMRoPE fix, check cos sim. If still <0.99, SSM formula differs → deeper investigation needed. |
| **Survivorship bias** | Only testing with 1-2 input tokens. Longer sequences may expose additional bugs. | Medium | After IMRoPE fix, test with 20+ token prompts and compare hidden states. |
| **IMRoPE implementation error** | Wrong section interleaving pattern produces wrong output. | Medium | Cross-check freq theta formula against llama.cpp reference. Test with single-token position encoding first. |
| **State staleness** | Walkway files from prior sessions had completely wrong claims. New files may also have errors. | High | Every claim now marked with verification status. Run DA sweep at end of each session. |

## Phase 3: Stale Claim Sweep Results

Files updated this session:
- `prestige_prompt.md`: v9→v10. Replaced all false ✅ claims with honest audit. Added IMRoPE spec.
- `goal-mantra.md`: v13→v14. Replaced Q5_K theory with IMRoPE root cause. Added compare command.
- `state.md`: v13→v14. Rewrote binary status table with honest da-v3 audit. Added cos sim diagnostics.
- `plan.md`: v11→v12. Replaced MoE/Q5_K theories with IMRoPE implementation plan.
- `overnight-map.md`: v2→v3. Updated workstreams to IMRoPE focus.

## Phase 4: Next Steps

### P0 — Implement IMRoPE
1. Add MROPE_SECTIONS[4] config parameter to model loading
2. Modify `precompute_rotary_kernel` in cuda_kernels.cu:
   - For each of 3 sections, compute sin/cos with different theta frequencies
   - theta_i = freq_base^(-2*section_offset[i]/ROTARY_DIM)
3. Modify `apply_rotary_qk_kernel` in cuda_kernels.cu:
   - Apply section-interleaved RoPE across head_dim
   - Only first ROTARY_DIM=64 dimensions (rest get 0 rotation)
4. Same changes for CPU GQA path in infer_text.c
5. Verify: cos sim > 0.99 vs ref

### When IMRoPE is Done
- Test essay generation: should produce coherent English
- Test 256K context: non-blocking test
- Run full test suite: every binary that loads the model
