# bytropix Plan — May 17 v17 (DA Audit Complete)

## STATUS
- **MoE**: ✅ cos-sim 1.0 internal consistency (lazy vs library, fresh build)
- **SSM**: ❌ L2 norm eps 1e-12 vs ~1e-6 (GGUF) — root cause of 0.006 gap
- **GQA**: ❓ not audited — needs DA pass
- **README/STATUS/prestige/overnight**: 🔴 all stale — need factual correction

## DA Audit: Phase 2 Pass 1
- [x] MoE lazy vs library: cos-sim 1.000000 verified on same input
- [x] Dequant bit-identity: confirmed across 8 top experts
- [x] Routing: identical top-8 selection
- [x] SSM recurrence formula: verified vs llama.cpp delta-net-base.cpp
- [x] L2 norm eps identified: 1e-12 (us) vs ~1e-6 (reference)

## Plan Forward (priority order)
1. [ ] Fix L2 epsilon: wubu_ssm.c:318-319 — read from GGUF instead of 1e-12f
2. [ ] Verify MOE=0 cos-sim vs reference after fix
3. [ ] Enable MOE=1, verify full model output
4. [ ] GQA path audit (10 layers, unverified)
5. [ ] Stale markdown cleanup (README, STATUS, prestige, overnight)
