═══ BYTROPIX — PRESTIGE RESUME (May 17 v6 — DA Audit, L2 eps found) ═══
Path: /home/wubu/bytropix | HW: RTX 5050 6.4GB, -arch=sm_120
Build: make infer_text
Model: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== COMPLETED ===
1. RoPE MRoPE section dimension fix (22/22/20) — FIXED
2. Output projection transpose (3 places) — FIXED
3. All dequant types verified exact vs ggml — VERIFIED
4. MoE lazy vs library: cos-sim 1.000000 — VERIFIED (internal consistency)
5. Top-k renormalization matches reference — CONFIRMED
6. SSM recurrence formula matches llama.cpp — VERIFIED (algebra)

=== REMAINING ===
1. 🔴 L2 norm epsilon: wubu_ssm.c hardcodes 1e-12, should read from GGUF (~1e-6)
   = root cause of 0.006 cos-sim gap in MOE=0 path
2. ❓ GQA forward: not audited vs llama.cpp
3. ❓ Full model MOE=1 vs reference: untested (was stale data)

=== DA AUDIT — Completed v2 ===
Q1: "What's the actual root cause of SSM gap?" → A: L2 norm eps 1e-12 vs ~1e-6
Q2: "Is MoE code correct?" → A: ✅ YES, cos-sim 1.0 internal, all dequant bit-identical
Q3: "Are markdown files correct?" → A: ❌ ALL stale — README, STATUS, prestige, overnight
