═══ GOAL PASTE (May 17 v22 — HONEST) ═══
PROJECT: bytropix — Custom Qwen3.6-35B-A3B inference engine
MODEL: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (mixed quant: IQ2_XXS/IQ3_XXS/Q5_K/Q6_K/F32)
STATUS: MoE contiguous dequant fixed. Output changed Chinese→English ("Doug").

=== FIXED THIS SESSION ===
- MoE expert weight layout: contiguous-per-expert (was reading garbage via interleaved)
- MOE default: 0→1
- MAX_LAYERS=0 clamp: fixed

=== REMAINING BUG (from research) ===
- **attn_output_gate: True** — official Qwen config. bytropix doesn't gate attention outputs with sigmoid(blk.X.attn_gate.weight). This is likely causing "Doug" vs ref "Here".

=== MODEL RESEARCH ===
- Architecture: 10x(3x Gated DeltaNet→MoE→1x GQA→MoE). DeltaNet = gated linear attention, NOT Mamba.
- "IQ2_M" is a label. Actual types: IQ2_XXS for experts, IQ3_XXS for down, Q5_K for attention, Q6_K for proj, F32 for small tensors.
- Unsloth Dynamic 2.0: selects quant types per-tensor by importance.

=== BUILD ===
cd /home/wubu/bytropix && make infer_text
=== TEST ===
NOGPU=1 MOE=1 MOE_LAYERS=0 ./infer_text /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Hello" 8 1
