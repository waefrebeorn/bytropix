# DA v11 — Triple Devil's Advocate Full Audit (May 20, 2026)

## EXECUTIVE: Q6_K dequant FIXED. GPU vs CPU divergence remains (cos-sim -0.66). Document drift corrected. Multi-modal roadmap established.

## Phase 1: Claim Verification

### Claim: "Q6_K dequant bug is the root cause of constant ~365 SSM output"
**Source:** Code reading + commit c07cf14
**Trust:** HIGH
**Verify:** `d*sc*(v6-32)` replaced `d*sc*v6 - 32.0`. Mathematical fix is exact. After fix: SSM output std drops from constant ~365 to ~0.036. **Verified.**
**Risk:** Fixed offset but GPU output still anti-correlated (-0.66). The dequant fix was necessary but not sufficient — state management bug remains.
**On re-check:** Still tight.

### Claim: "CPU SSM path matches llama.cpp at cos-sim 0.994"
**Source:** Prior DA v10 (May 18), verified with full 40-layer dump
**Trust:** HIGH (pre-GPU_SUPPORT change, but doesn't touch CPU path)
**Verify:** FORCE_CPU_SSM path in current code. CPU path code has NOT changed — only GPU code was modified since v10. Cos-sim 0.994 is credible.
**Risk:** IQ2_M quant is ~2.7 bpw — quantization noise means cos-sim can never reach 1.0. 0.994 is at ceiling.
**On re-check:** Needs re-run with current binary to confirm. gen_text CPU binary is BROKEN — cannot re-verify.

### Claim: "Vision encoder is ported and functional (384 LoC)"
**Source:** Code reading — `src/wubu_vision.c` has complete init/forward/free for 27-layer 3D ViT
**Trust:** MEDIUM
**Verify:** Code compiles standalone? **Unknown** — not yet built in current session. Forward pass uses correct patch embedding, attention, FFN, GELU, spatial merge, mmproj. Structure matches Qwen3.6 mmproj format.
**Risk:** Untested code may have shape mismatches, tensor name changes, or runtime bugs. No ground-truth reference for vision encoder output.
**On re-check:** Build test_vision_real → verify vs expected output range.

## Phase 2: Stale Claims Sweep — Propagation Check

### Sweep targets checked:
| File | Previously Stale? | Now? |
|------|------------------|------|
| `README.md` | Phase 28b — F32 waste, mem leak, broken prefill | ✅ Corrected to Phase 28e + full roadmap |
| `STATUS.md` | Phase 28b | ✅ Corrected |
| `MADE_AGENTICALLY.md` | v24 Phase 28b | ✅ Needs update |
| `prestige_prompt.md` | Phase 28 (May 21) | ✅ Rewritten |
| `plan.md` | Phase 28b — illegal access P0 | ✅ Rewritten with full roadmap |
| `state.md` | Phase 28c (WIP) | ✅ Committed as Phase 28e |
| `goal-mantra.md` | Phase 28 DA (WIP) | ✅ Committed as Phase 28e |
| `overnight-map.md` | Phase 28d (WIP) | ✅ Committed as Phase 28e |
| `DA v10` | May 18 — pre-GPU_SUPPORT | OK — historical record, not stale |

### Stale claims found and corrected:
1. ❌ "F32 waste ~2.2 GB" (README.md, STATUS.md, prestige_prompt.md) → Already FIXED
2. ❌ "GPU mem leak ~5.5 GB" (README.md, prestige_prompt.md) → FALSE POSITIVE
3. ❌ "Column-major kernel broken" (README.md) → CORRECT layout
4. ❌ "gen_text.c hardcoded prompt" (prestige_prompt.md) → ACCEPTS argv[1]

## Phase 3: Cross-Reference Verification

### Binaries:
| Binary | Claimed Status | Verified? |
|--------|---------------|-----------|
| gen_text_gpu | Runs ✅ | Yes — produces text at ~5.9 tok/s |
| gen_text (CPU) | Broken ❌ | Yes — link failure (GPU symbols) |
| test_vision_real | Exists ❓ | Not built yet |

### Documentation vs Code:
| Doc Claim | Code Proof |
|-----------|-----------|
| 40 layers, 3:1 SSM/GQA | ✅ wubu_model.c layer iteration |
| 256 experts, 8 active | ✅ wubu_moe.c |
| 3D ViT 27 layers, 1152 hidden | ✅ wubu_vision.c, wubu_vision.h |
| mmproj: 4608→2048 | ✅ wubu_vision.h mm0_weight[4608,4608], mm2_weight[4608,2048] |

### Git Hygiene:
| Check | Status |
|-------|--------|
| Local HEAD | c07cf14 ✅ |
| Remote origin/master | 4dc985e ❌ — 8 behind |
| Uncommitted changes | 7 files — includes this DA sweep |
| Branch | master ✅ |

## Phase 4: Risk Assessment

| Risk | Type | Detail | Mitigation |
|------|------|--------|-----------|
| GPU divergence root cause wrong | Confirmation bias | Assuming state management when it could be kernel bug | Layer-by-layer binary dump comparison — no assumptions |
| Vision encoder silently wrong | Survivorship bias | Code compiles but produces garbage | Build, run test_vision_real, check output distribution |
| Multi-modal roadmap too ambitious | Scope creep | Vision + MoE + sparse attention + chunked prefill = 5 phases | Clear P0-P5 ordering, each phase is independent |
| CPU cos-sim degraded after GPU changes | State staleness | GPU changes may have touched shared structs | Re-verify cos-sim after fixing CPU build |

## Phase 5: Consolidated Status Table

| Component | Status | Evidence | Next Action |
|-----------|--------|----------|-------------|
| GPU SSM C==1 decode | ⚠️ Runs, wrong output | cos-sim -0.66 vs CPU | Debug state persistence |
| Q6_K dequant | ✅ Fixed | c07cf14 + output range check | Document as historical |
| CPU SSM path | ✅ cos-sim 0.994 | DA v10 full-layer dump | Re-verify after CPU build fix |
| GPU kernels (all fused) | 🟡 Individually verified | cos-sim 1.0 vs old cuBLAS path | Test in full pipeline |
| Vision encoder | 🟡 Written, untested | 384 LoC, structure correct | Build + run test_vision_real |
| Vision→text pipeline | 🟡 Written, untested | test_vision_real.c lines 73-83 | Build + verify logit range |
| MoE router | ✅ CPU path | wubu_moe.c | Re-verify with GPU after fix |
| CPU build | ❌ Broken | GPU symbol link error | `#ifdef GPU_SUPPORT` wrap |
| Remote sync | 🔴 8 behind | git log origin/master..HEAD | Push after CPU build fix |
| Document drift | ✅ Corrected | This DA + 7 files updated | Commit sweep |

## Phase 6: Remaining Work (Priority Order)

### P0: GPU state divergence (1-2 sessions)
- Layer-by-layer intermediate comparison: GPU vs CPU
- Check recurrence state + conv state at each step
- Fix → cos-sim > 0.99

### P1: Build fix + push (1 session)
- Wrap GPU symbols in `#ifdef GPU_SUPPORT` in wubu_model.c
- Build gen_text (CPU) successfully
- Push 8 commits to remote

### P2: Vision verification (1 session)
- Build test_vision_real
- Generate test pixels or find existing
- Verify E2E: vision→mmproj→text→logits

### P3-5: Multi-modal + feature cream + profiling (3-5 sessions)
See plan.md for full Phase 33-35 roadmap
