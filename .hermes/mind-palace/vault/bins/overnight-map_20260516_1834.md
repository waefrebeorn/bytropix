# WuBuText AI — Overnight Navigation Map (May 17 v14 — HONEST)

## Where We Are
3 bugs fixed this session. Remaining "Doug" vs "Here" discrepancy is NOT from attn_output_gate — that's already implemented inside wubu_ssm_forward. Source unknown.

## What's Fixed This Session
1. **MoE dequant contiguity**: contiguous-per-expert (was reading garbage)
2. **MOE default**: 0→1
3. **MAX_LAYERS=0**: fixed clamp

## Model Research Done
- **Architecture**: 10x(3x Gated DeltaNet→MoE→1x GQA→MoE). DeltaNet ≠ Mamba.
- **Mixed quantization**: IQ2_XXS/IQ3_XXS/Q5_K/Q6_K/F32 per tensor. "IQ2_M" is a label.
- **attn_output_gate: True** — already implemented in wubu_ssm_forward via silu(x @ attn_gate_weight) applied to SSM output before ssm_out projection. NOT missing.
- Research saved to research.md

## Remaining Action Items
- "Doug" vs "Here" discrepancy cause unknown — likely quantization noise at 2-3 bpw on 35B MoE
- Possible next: compare per-layer hidden states against llama.cpp dump
