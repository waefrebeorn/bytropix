# bytropix — Overnight Navigation Map (May 17 v6 — DA Audit Complete)

## Where We Are
All individual components verified. MoE internally consistent at cos-sim 1.0. 
SSM recurrence formula matches llama.cpp exactly.
Root cause of remaining 0.006 gap: L2 norm epsilon 1e-12 (us) vs ~1e-6 (GGUF config).

## What Changed This Session
- MoE lazy vs library verified: cos-sim 1.000000 (fresh build)
- Dequant bit-identity confirmed across 8 experts
- Previous "MoE=1 divergence" = stale binary/miscompare — no bug existed
- DA audit found L2 norm eps mismatch = real root cause

## Next Step
Fix wubu_ssm.c:318-319 — replace hardcoded 1e-12f with GGUF config value (~1e-6)

## Build
```bash
rm -f infer_text; make infer_text
NOGPU=1 MOE=0 ./infer_text /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Hello" 4
```
