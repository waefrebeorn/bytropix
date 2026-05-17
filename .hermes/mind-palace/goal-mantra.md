═══ GOAL PASTE (May 17 v17 — DA Audit Complete) ═══
PROJECT: bytropix — 1:1 parity with llama-cli
MODEL: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== STATUS ===
- MoE: ✅ cos-sim 1.0 internal (lazy vs library match)
- SSM: ❌ L2 norm eps 1e-12 vs ~1e-6 (GGUF config) = 0.006 gap in MOE=0
- GQA: ❓ not audited
- README: 🔴 stale — claims "MoE stride bug" (disproven)

=== FIX ORDER ===
1. Fix L2 epsilon: wubu_ssm.c line 318-319, read from model GGUF
2. Verify MOE=0 cos-sim ~1.0 vs reference
3. Enable MOE=1, verify cos-sim ~1.0 vs reference
4. Audit GQA path
5. Fix all stale markdown files

=== BUILD ===
make infer_text
NOGPU=1 MOE=0 ./infer_text /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Hello" 4
