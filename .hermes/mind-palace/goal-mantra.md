# Goal Mantra — May 18, 2026 — POST-FIX

## THE GOAL
1:1 logit parity with llama.cpp for Qwen3.6-35B-A3B-UD-IQ2_M.
Cos-sim 0.9968 achieved. Next: push to 0.999+.

## ACHIEVED THIS SESSION
- **GQA Q/gate interleave bug FIXED**: attn_q.weight output [8192] is per-head
  interleaved as [Q_h0(256)][gate_h0(256)][Q_h1(256)][gate_h1(256)]...
  Our code split into two contiguous blocks — WRONG. Single fix raised
  cos-sim -0.51 → 0.9968. This was THE bug blocking inference for weeks.

- **MoE quantized path WIRED**: IQ2_XXS/IQ3_XXS/IQ4_XS vec_dot → blob pointers
  → quantized_matmul for both shared and routed experts.

- **Per-layer dump INFRASTRUCTURE**: Modified llama.cpp to dump per-layer hidden
  states via LLAMA_DUMP_LAYERS=1 + DUMP_LAYER_DIR. Same env var works for our model.

## REMAINING GAP
Cos-sim 0.9968 vs 1.0. Source: quantized_matmul's input Q8_K quantization
+ self-contained C vec_dot differ from llama.cpp's SIMD paths.
Each layer accumulates ~0.0003 quantization noise. ALL 40 layers > 0.995.
This is acceptable for IQ2_M (2.7 bpw) quantization level.

## GROUND TRUTH
- Reference: ~/llama.cpp/src/models/qwen35moe.cpp
- GGUF metadata: arch="qwen35moe", ssm_n_group=16, ssm_dt_rank=32, ssm_d_state=128
- Reference binary: ~/llama.cpp/build/bin/llama-cli
- Hidden state per-layer dump: DUMP_LAYER_DIR=/tmp/dump_layers env var

## REAL VERIFICATION
1. Run both ref and our model with DUMP_LAYER_DIR set
2. python3 to compare layer-by-layer cos-sim
3. All layers must have cos-sim > 0.99 (achieved: > 0.995)
