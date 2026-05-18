# Prestige Prompt — May 18, 2026 — POST-FIX

## Project: bytropix — Qwen3.6-35B-A3B-UD-IQ2_M
Cos-sim vs ref: **0.9968** (up from -0.51). All 40 layers > 0.995.

## Architecture (qwen35moe → qwen35moe.cpp)
40 layers: 30 SSM (Gated DeltaNet) + 10 GQA (full attention)
- L0,1,2 → L3 GQA → L4,5,6 → L7 GQA → ... → L39 GQA
- Hidden: 2048, Vocab: 248320, Expert dim: 512, Shared dim: 512
- SSM: 16 K-heads, 32 V-heads, d_state=128, conv_kernel=4
- GQA: 16 Q-heads, 2 KV-heads, head_dim=256

## CRITICAL BUG FIXED THIS SESSION
GQA Q/gate extraction from attn_q.weight was WRONG. Reference (qwen3next.cpp) 
uses per-head interleaved layout [Q_h0][gate_h0][Q_h1][gate_h1]... 
Our code split into [Q_all][gate_all] — two contiguous blocks.
Fix: per-head interleaved extraction for both gate copy and Q normalization.

Single fix: cos-sim -0.51 → 0.9968. Years-long inference bug resolved.

## What's Wired
- SSM: quantized_matmul (Q5_K/Q6_K) ✓
- GQA: quantized_matmul (Q5_K) ✓ with interleave fix
- MoE shared expert: quantized_matmul (Q5_K/Q6_K) ✓
- MoE routed experts: quantized_matmul (IQ2_XXS/IQ3_XXS/IQ4_XS) ✓
- Router + gate: F32 from blob ✓
- Output: Q4_K quantized matmul ✓
- Per-layer dump via DUMP_LAYER_DIR env var ✓

## Per-Layer Dump Tools
- Reference: LD_LIBRARY_PATH=~/llama.cpp/build/bin LLAMA_DUMP_LAYERS=1 DUMP_LAYER_DIR=/tmp/dump_layers /tmp/llama_dump
- Our model: DUMP_LAYER_DIR=/tmp/dump_layers ./test_full_moe
- Compare: python3 script computes cos-sim per layer

## Unsloth type verification (actual GGUF scan)
NOT "IQ2_XS/IQ3_XS/Q3_S_XL" as earlier markdown claimed.
Actual: IQ2_XXS(type16,80), IQ3_XXS(type18,37), IQ4_XS(type23,3).
Vec_dot for actual types exists and is wired.
