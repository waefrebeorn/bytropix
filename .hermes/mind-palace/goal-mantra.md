═══ GOAL PASTE (May 17 v16 — All components verified, MoE divergence = recurrent amp) ═══
PROJECT: bytropix — Custom Qwen3.6-35B-A3B inference engine
MODEL: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
┌─────────────────────────────────────────────────────────────┐
│ ARCH: qwen35moe │ 40L │ D_MODEL=2048 │ Q16/KV2 │ SSM16/32 │
│ MoE: 256 experts │ top-8 │ shared │ IQ2_XXS gate/up/down │
└─────────────────────────────────────────────────────────────☆

=== FIXED ===
1. RoPE MRoPE section dimension — was ROTARY_DIM=64, now per-section 22/22/20
   cos-sim -0.456 → -0.016 (MOE=0). Then verified SSM/GQA correct at cos-sim 0.994
2. MoE expert layout NOT interleaved — dims[0] innermost, experts contiguous
   dequant_multi_expert_contiguous is correct. All dequant types verified vs ggml:

| Type | Tensor | Verification |
|------|--------|-------------|
| IQ2_XXS (16) | gate/up exps | full 1M match exp0+64 |
| IQ3_XXS (18) | down exps (L0-36) | full 1M match exp0+64 |
| IQ4_XS (23) | down exps (L37-39) | full 1M match exp0 |
| Q5_K (13) | sh gate/up | 256/256 match |
| Q6_K (14) | sh down | loaded via gguf_read_tensor_f32 |
| Q5_K embd | token_embd (BOS) | 2048/2048 match |

=== REMAINING ===
3. MoE=1 vs reference cos-sim 0.337 logits — divergence from recurrent amplification
   SSM/GQA path correct (cos-sim 0.994 MOE=0 logits)
   Every subcomponent verified exact vs ggml
   Hypothesis: tiny per-layer MoE differences compound through 40 SSM layers
4. Next: compare per-layer MOE=1 vs MOE=0 to find divergence onset layer

=== TOOLS ===
- infer_text: main inference (NOGPU=1 for CPU, MOE=0|1)
- run_ref_moe0: reference llama.cpp via API (final hidden + logits only)
- dump_full_layers: library-only forward (pass-through FFN, no MoE)
- compare_weights: dump dequantized tensor from GGUF

=== BUILD ===
rm -f infer_text dump_full_layers run_ref_moe0 compare_weights; make infer_text
gcc -O2 -I include -o dump_full_layers tools/dump_full_layers.c \
    src/gguf_reader.o src/wubu_ssm.o src/wubu_mobius.o \
    src/wubu_moe.o src/wubu_model.o src/wubu_tokenizer.o \
    src/qlearner.o -lm -fopenmp
