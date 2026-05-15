# Unsloth Dynamic GGUF Format & Qwen3.6-35B-A3B Architecture

## Source: Unsloth Documentation & Hugging Face config.json
- Unsloth docs: https://unsloth.ai/docs/models/qwen3.6
- HF config: Qwen/Qwen3.6-35B-A3B

## UD (Unsloth Dynamic) Quantization

**What "UD" means:** Per-tensor dynamic quantization — each weight tensor gets the quantization type that best preserves its accuracy. Unlike standard GGUF where all tensors use same quant, UD mixes types.

**Filename convention:**
- `UD-Q4_K_XL` = Dynamic Q4_K (4-bit for most weights, higher precision for critical tensors)
- `UD-Q2_K_XL` = Dynamic Q2 (2-bit)
- `UD-IQ2_M` = Dynamic IQ2 medium (our model)

**Per-tensor quantization in Qwen3.6-35B-A3B-UD-IQ2_M:**
- `token_embd.weight`: Q5_K (type 13) — 5.56 bpw
- `output.weight`: Q6_K (type 14) — 6.56 bpw
- `attn_qkv.weight` (SSM layers): Q5_K (type 13)
- `attn_gate.weight` (SSM layers): Q5_K (type 13)
- `attn_q.weight`, `attn_k.weight`, `attn_v.weight` (GQA layers): Q5_K (type 13)
- `ssm_*` weights: F32 (type 0) — full precision for small params
- `ffn_gate_exps.weight`, `ffn_up_exps.weight`: IQ2_XXS (type 16) — 2.06 bpw
- `ffn_down_exps.weight`: IQ3_XXS (type 18) — 3.44 bpw
- `ffn_gate_shexp.weight`, `ffn_up_shexp.weight`: Q5_K (type 13)
- Norms/biases: F32 (type 0)

**Key insight:** The "IQ2_M" in the filename describes the MoE expert weights (dominant by parameter count), but critical tensors (embeddings, QKV projections) use higher precision Q5_K/Q6_K.

**llama.cpp is the reference inference engine.** Unsloth recommends building llama.cpp from source for UD-GGUF models.

## Qwen3.6-35B-A3B Architecture (from config.json)

**Model type:** `qwen3_5_moe` — Hybrid Gated DeltaNet + GQA + MoE

### Core Parameters
- `hidden_size`: 2048
- `num_hidden_layers`: 40
- `head_dim`: 256
- `vocab_size`: 248320
- `max_position_embeddings`: 262144
- `rms_norm_eps`: 1e-06
- `hidden_act`: `silu` (SwiGLU)
- `tie_word_embeddings`: false
- `bos_token_id`: 248044
- `eos_token_id`: 248044 (SAME as BOS!)

### Hybrid Attention (30 SSM + 10 GQA)
- `full_attention_interval`: 4 (every 4th layer is GQA)
- `layer_types`: 3× linear_attention, 1× full_attention (repeated 10× = 40 layers)
- `attn_output_gate`: true

**SSM (Gated DeltaNet) layers (30 of 40):**
- `linear_num_key_heads`: 16 (SSM_K_HEADS)
- `linear_num_value_heads`: 32 (SSM_V_HEADS)
- `linear_key_head_dim`: 128 (SSM_D_STATE)
- `linear_value_head_dim`: 128
- `linear_conv_kernel_dim`: 4 (CONV_KERNEL)
- `ssm_time_step_rank`: 32 (DT_RANK)
- `ssm_inner_size`: 4096 (VALUE_DIM = SSM_V_HEADS × SSM_D_STATE)
- Uses `attn_gate.weight` for output gate (sigmoid gate after SSM recurrence)

**GQA layers (10 of 40):**
- `num_attention_heads`: 16 (GQA_Q_HEADS)
- `num_key_value_heads`: 2 (GQA_KV_HEADS)
- Q weight is fused with gate: [D_MODEL, Q_HEADS×HEAD_DIM×2] = [2048, 8192]
- Gate is sigmoid applied to attention output before output projection

### MoE
- `num_experts`: 256
- `num_experts_per_tok`: 8
- `moe_intermediate_size`: 512 (D_FF)
- `shared_expert_intermediate_size`: 512 (SHARED_D_FF)
- `router_aux_loss_coef`: 0.001

### RoPE
- `rope_theta`: 10000000.0
- `partial_rotary_factor`: 0.25 (64 dims out of 256 are rotated)
- `rope_type`: "default" (standard RoPE, NOT MRoPE despite mrope_section being defined)
- `mrope_section`: [11, 11, 10] (for vision encoder, not text-only)

### Tokenizer
- `tokenizer.ggml.add_bos_token`: false (but model requires BOS for correct inference)
- BPE tokenizer with GPT-2 pre-tokenizer
- 247,587 merges

### MTP (Multi-Token Prediction)
- `mtp_num_hidden_layers`: 1 (for speculative decoding)
- `mtp_use_dedicated_embeddings`: false

### Vision (unused in text-only mode)
- `vision_config.depth`: 27
- `vision_config.hidden_size`: 1152
- `vision_config.out_hidden_size`: 2048 (projects to text space)

## Known Implementation Issues (Current Codebase)

### Critical (model produces garbage output)
1. **CPU GQA missing output gate** — `infer_text.c` GQA forward doesn't apply `sigmoid(gate) * attn_out`. SSM layers have the gate, GQA layers don't (in CPU path).
2. **Q5_K dequant block-level identical values** — 56/64 blocks of 32 elements are constant. Either a dequant bug or real Q5_K artifact. Old and new dequant code produce same output for this specific tensor.
3. **EOS=BOS (248044)** — Need to handle stopping condition carefully.

### Moderate
4. **BOS prepending** — `add_bos_token: false` in config, but model still needs BOS for correct completions.
5. **No chat template** — Qwen3.6 is instruction-tuned. Needs `<|im_start|>` chat format or raw continuation won't work well.
6. **Greedy decoding only** — No temperature/top-k/top-p/min-p/presence-penalty. Unsloth recommends `--temp 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5 --min-p 0.00`.

### Future (not blocking)
7. **MTP head** — Not used in inference; only needed for speculative decoding speedup.
8. **Vision encoder** — Only needed for multimodal, not text-only.
9. **GGUF tensor names match code** — Verified: blk.{N}.attn_qkv.weight, attn_gate.weight, ssm_*.weight all correct.
10. **Output weight shape** — GGUF stores [D_MODEL, vocab_size] = [2048, 248320], flattened to row-major for SGEMM. Confirmed correct.

## Recommended Verification
1. Build llama.cpp and run the same model to get reference output
2. Compare hidden states at layer boundaries between our code and llama.cpp
3. Check CPU GQA gate application
