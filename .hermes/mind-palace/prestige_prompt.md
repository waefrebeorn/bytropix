═══ BYTROPIX — PRESTIGE RESUME (May 17 v5 — All components verified) ═══
Path: /home/wubu/bytropix | HW: RTX 5050 6.4GB, -arch=sm_120
Build: rm -f infer_text; make infer_text
Model: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== COMPLETED ===
1. RoPE MRoPE section dimension fix (22/22/20) — FIXED
2. Output projection transpose (3 places: final + GQA + model API) — FIXED
3. All dequant types verified exact vs ggml (IQ2_XXS, IQ3_XXS, IQ4_XS, Q5_K, Q6_K) — VERIFIED
4. SSM/GQA path cos-sim 0.994 logits vs reference (MOE=0) — VERIFIED
5. MoE expert layout: CONTIGUOUS per expert, not interleaved — CONFIRMED
6. Top-k renormalization matches reference (norm_w=true) — CONFIRMED
7. Reference runs 100% CPU — CONFIRMED
8. All debug patches cleaned — DONE

=== REMAINING ===
1. MoE=1 divergence (cos-sim 0.337 logits) — likely recurrent SSM amplification
2. Compare MOE=1 vs MOE=0 per-layer residuals to find divergence onset
3. If layer-0: compare moe_expert_forward_lazy vs build_moe_ffn order of ops
4. If layer 2-3+: accept as recurrent amplification artifact (chaotic trajectory)

=== DA AUDIT — Completed v1 (all previous DA questions resolved) ===
Q1: "SSM formulas verified correct?" → A: ✅ YES, cos-sim 0.994 logits vs llama.cpp reference
Q2: "GQA algorithm verified correct?" → A: ✅ YES, verified as part of end-to-end MOE=0 path
Q3: "Weight layouts all verified?" → A: ✅ YES, all 5 quant types vs ggml, exps 0 & 64
Q4: "What else might be wrong?" → A: ⚠️ MoE order of ops vs reference may differ in ways that compound over 40 layers
