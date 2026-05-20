# Plan Forward: True 1:1 Parity

**Current:** 0.9994 overall cos-sim. L31 at 0.9585.
**Target:** Every layer > 0.99, logits match llama.cpp within known quantization noise.

## The L31 Problem

L31 is a GQA-only layer (index 3 in every 4-cycle). Its cos-sim drops to 0.9585. This is NOT quantization noise in the traditional sense — it's **error amplification through 30 layers of interleaved SSM+GQA before hitting a pure GQA layer**.

### Root Cause Analysis

The chain of error:

1. L00-L30: Each layer adds ~0.06% quantization noise from different quantized matmul implementations (bytropix Q8_K→vec_dot vs llama.cpp ggml_vec_dot)
2. L30 output cos-sim: 0.9994 (cumulative 0.06% error)
3. L31 is pure GQA: its attention mechanism is SENSITIVE to subtle differences in Q/K values
4. The attention softmax amplifies small Q·K differences: a 0.06% Q difference can produce a 4% attention weight shift
5. Result: L31 output cos-sim drops to 0.9585

### Fix Strategy (P0-P1)

**P0 — Immediate: Use DUMP_INTERMEDIATE_DIR to compare L31 Q, K, V values**

The reference intermediate dumps contain L31_Qcur.bin, L31_Kcur.bin, L31_Vcur.bin. Bytropix needs to dump the same intermediates. Add `DUMP_GQA_DEBUG_DIR` to `wubu_gqa_forward()` that saves Q_full, K, V before and after RMSNorm, and after RoPE.

Then compare:
- bytropix L31_Qcur vs ref L31_Qcur → if match, problem is in attention
- bytropix L31_Kcur vs ref L31_Kcur → if mismatch, problem is in K projection
- bytropix L31_attn_norm vs ref L31_attn_norm → if mismatch, problem is in RMSNorm

**P1 — If Q/K match but attention diverges: compare attention score distributions**

The reference has L31___fattn__.bin (attention scores). Compare bytropix's raw attention scores vs reference. If they diverge, the online softmax or score computation is the cause.

**P1 — If Q/K projections diverge: check quantized matmul**

Test the Q5_K matmul for L31 against F32 SGEMM. If bytropix's matmul produces different results than llama.cpp's, the issue is in `quantized_dot_generic.c` (Q5_K vec_dot).

## Implementation Plan

### Step 1: Add DUMP_GQA_DEBUG_DIR to wubu_gqa_forward() (2 hours)
Add ENV var check at function entry. Save Q_full, K, V, Q_norm, K_norm, attn_weights to files. Same format as DUMP_INTERMEDIATE_DIR.

### Step 2: Run comparison for L31 (30 min)
```bash
DUMP_GQA_DEBUG_DIR=/tmp/gqa_ref DUMP_GQA_LAYER=31 ./ref_dumper model.gguf "prompt" 0
DUMP_GQA_DEBUG_DIR=/tmp/gqa_our DUMP_GQA_LAYER=31 ./gen_text "prompt" 0
python3 tools/compare_gqa_intermediates.py /tmp/gqa_ref /tmp/gqa_our
```

### Step 3: Fix divergence found (1-4 hours)
- If Q5_K matmul: fix quantized_dot_generic.c Q5_K vec_dot
- If softmax: fix wubu_ssm.c softmax implementation
- If RoPE: fix IMRoPE frequency computation
- If KV cache read: fix kv_cache_read_head Q4_0 dequant

### Step 4: Re-verify (30 min)
```bash
DUMP_LAYER_DIR=/tmp/ref ./ref_dumper model.gguf "prompt" 0
DUMP_LAYER_DIR=/tmp/our ./gen_text "prompt" 0
tools/layer_cos_sim /tmp/ref /tmp/our 40
```

## Expected Outcome
- L31 cos-sim > 0.99 (from 0.9585)
- Overall cos-sim > 0.9995
- All 40 layers > 0.99

## If the Gap Persists
If fixing L31 still leaves a gap, the remaining 0.0006 difference is from bytropix vs llama.cpp quantized matmul implementation differences. These are:
- Q5_K vec_dot (SSM attn_qkv, attn_gate, GQA Q/K/V/output)
- Q6_K vec_dot (SSM ssm_out, shared expert down)
- Q4_K vec_dot (output projection)

To close this fully, we would need to match llama.cpp's exact vec_dot implementation (which uses specific AVX2 instruction ordering and FMA accumulation patterns). This is inherently fragile — different compiler versions will produce different ordering.

**Practical threshold**: 0.999+ overall with no layer below 0.99 is sufficient for tool-call accuracy. The difference between 0.9995 and 1.0 is below the quantization noise floor.
