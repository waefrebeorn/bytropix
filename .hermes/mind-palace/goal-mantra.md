# Goal Mantra — May 18, 2026 — HONEST REBOOT

## THE REAL GOAL
Fix SSM/GQA forward architecture. Model produces garbage even with ALL-F32 math.
Cos-sim -0.128 without MoE, all quantized path disabled. Architecture bug, not quantization.

## GROUND TRUTH
- Reference: ~/llama.cpp/src/models/qwen3next.cpp (NOT qwen35moe.cpp)
- GGUF says "qwen35moe" but llama.cpp routes to LLM_ARCH_QWEN35MOE → qwen35moe.cpp
- qwen35moe SSM forward is copied from qwen3next.cpp (same build_layer_attn_linear)
- Weight layout: qwen35moe-style (separate ssm_beta/ssm_alpha, not fused)
- Formula reference: qwen3next.cpp build_layer_attn_linear, build_delta_net_autoregressive

## WHAT WE KNOW
- SSM Layer 0 cos_sim=0.40 vs ref (from prev sessions)
- GQA layers (3,7,11,15,19,23,27,31,35,39) use separate Q/K/V weights
- attn_q.weight [2048,8192] = Q[4096] + gate[4096] fused — split needed
- Q6_K dequant block layout had bugs (may still be wrong in quantized_dot_generic.c)
- IMRoPE is a no-op for text-only (all positions = token_pos)

## WHAT WAS DONE THIS SESSION
- MoE quantized path wired via blob pointers (shared + routed experts)
- MoE weights use IQ2_XXS/IQ3_XXS/IQ4_XS vec_dot (correct types verified)
- Cos-sim still -0.65 with quantized MoE (expected — SSM/GQA layers are wrong)

## REAL NEXT STEPS (not the wrong plan from before)
1. Layer-by-layer comparison: modify llama.cpp to dump per-layer hidden states
2. Find first diverging layer (expected: Layer 0 SSM)
3. Fix SSM forward: conv1d, delta-net recurrence, gating
4. Fix GQA forward: separate Q/K/V weight split + attention
5. Re-test with quantized MoE (should work once hidden states correct)

## NO CHOICES
Never present options. Never say "this would take X". Never say "too much work".
Just do it. Loop until done.
