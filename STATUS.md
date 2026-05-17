═══ BYTROPIX — STATUS (May 17 — DA Audit: SSM L2 eps = root cause) ═══
Path: /home/wubu/bytropix | Branch: master
HW: RTX 5050 6.4GB, -arch=sm_120

=== INFERENCE PARITY STATUS (C code to C code) ===
| Component | vs llama.cpp | Notes |
|-----------|-------------|-------|
| MoE lazy forward | ✅ cos-sim 1.0 | Decides to t+library, all dequant bit-identical |
| SSM L2 norm eps | ❌ 1e-12 vs ~1e-6 | Hardcoded in wubu_ssm.c:318-319 |
| SSM recurrence | ✅ Formula matches | Verified vs llama.cpp delta-net-base.cpp |
| GQA | ❓ Not audited | 10 layers, untested since output proj transpose fix |
| Output proj | ✅ Fixed | 3 transpose fixes, verified |
| RoPE MRoPE | ✅ Fixed | sections 22/22/20 (was 64) |

=== FIX ORDER ===
1. Fix L2 epsilon (wubu_ssm.c:318-319)
2. Verify MOE=0 cos-sim vs ref
3. Verify MOE=1 cos-sim vs ref
4. Audit GQA
5. Clean stale docs
