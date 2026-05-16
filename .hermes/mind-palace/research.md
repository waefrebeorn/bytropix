# Qwen3.6-35B-A3B-UD-IQ2_M Research

## 1. Model Creator: Qwen (Alibaba)
**Source:** https://huggingface.co/Qwen/Qwen3.6-35B-A3B

**Architecture (qwen3_5_moe):**
- 35B total params, 3B active per token
- 40 layers: **10x (3x Gated DeltaNet -> MoE -> 1x Gated Attention -> MoE)**
- Gated DeltaNet (SSM): 32 V-heads, 16 QK-heads, hd=128, conv_kernel=4, mamba_ssm_dtype=float32
- Gated Attention (GQA): 16 Q-heads, 2 KV-heads, hd=256, RoPE dim=64 (25% partial), theta=10M
- **attn_output_gate: True** — EVERY attention output has a sigmoid gate (`blk.X.attn_gate.weight`). bytropix MISSING this!
- MoE: 256 experts, 8 routed + 1 shared, expert_dim=512, shared_dim=512
- Hidden: 2048, vocab: 248320 (padded)
- Context: 262K native (up to 1,010,000)
- MTP: 1 extra prediction head
- activ: silu, norm_eps: 1e-6

## 2. GGUF Provider: Unsloth
**Source:** https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF

**Unsloth Dynamic 2.0 Quantization:**
- NOT one type — selects different quant types per tensor based on importance
- Newer than standard imatrix: model-specific layer selection, 1.5M+ token calibration
- Claims better KLD / Aider than standard quants despite 8GB smaller
- "UD" = Unsloth Derivatives — standard GGUF, compatible with llama.cpp

## 3. Actual Quantization Mix
| Type | Count | Used For |
|------|-------|----------|
| F32 | 361 | Router, norms, biases, small projections |
| Q5_K | 181 | Attention QKV, shared expert gate, attn_gate |
| IQ2_XXS | 80 | MoE gate_exps, up_exps (routed experts) |
| Q6_K | 70 | SSM output proj, some attention |
| IQ3_XXS | 37 | MoE down_exps (routed expert down proj) |
| IQ4_XS | few | Some layers' down_exps (e.g. L39) |

"IQ2_M" in filename is a **label** for overall compression, not a GGML type.

## 4. Layer Types
- SSM (Gated DeltaNet): L0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38 (30)
- GQA: L3,7,11,15,19,23,27,31,35,39 (10)

## 5. Action Items for bytropix
- **[RESOLVED] attn_output_gate** — already implemented inside wubu_ssm_forward via silu(x @ attn_gate_weight). Applied to SSM raw output before ssm_out projection. GQA layers don't have attn_gate tensors.
- **[DONE] Down_exps type varies per layer** (some IQ4_XS) — already handled via ty_gd fallback
- **[DONE] Expert weights are contiguous-per-expert** — our fix was correct
