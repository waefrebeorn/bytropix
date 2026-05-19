═══ BYTROPIX — STATUS (May 19 — Phase 9.5: Q6_K FIXED — 1:1 PARITY) ═══
Path: /home/wubu/bytropix | Branch: master
HW: AMD Ryzen 7950X 16C/32T, 64GB DDR5, RTX 5050 6.4GB
Model: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (2.7 bpw, 10.7 GB)

=== INFERENCE PARITY (C code to C code) ===
| Component            | Status          | Notes                              |
|----------------------|-----------------|------------------------------------|
| Full 40L forward     | ✅ cos-sim 0.9967| Q6_K vec_dot bug fixed!            |
| GQA Q/gate interleave| ✅ FIXED        | cos-sim -0.51 → 0.9968             |
| IMRoPE               | ✅ Implemented   | sections=[11,11,10,0], theta=10M   |
| MoE quantized path   | ✅ Wired         | IQ2_XXS/IQ3_XXS/IQ4_XS via blob    |
| Shared expert gate   | ✅ Sigmoid gate  | ffn_gate_inp_shexp * sigmoid(x_s)  |
| Q5_K quant matmul    | ✅ cos-sim 0.9999| Verified vs F32 SGEMM               |
| Q6_K quant matmul    | ✅ cos-sim 0.9999| **FIXED** was 0.728 — loop iter bug|
| MoE router gating    | ✅ Softmax       | identical to llama.cpp qwen35moe   |
| Output proj Q4_K     | ✅ cos-sim 0.99995| vs F32 SGEMM                       |
| Decode speed         | ⚠️ 4.7 tok/s     | CPU-only, 16 threads               |
| Prefill speed        | ℹ️ 16.2 tok/s    | 27-token prompt                    |
| Chat template        | ❌ Not applied   | Minor quality impact               |

=== ROOT CAUSE FIXED ===
The 0.794 cos-sim was caused by a Q6_K AVX2 vec_dot loop iteration bug:
- quantized_dot_generic.c:314 — `j < QK_K/32` → `j < QK_K/16`
- Only 128/256 elements were processed per block (50% coverage)
- Impacted shared expert output projection across ALL 40 layers
- Previous "softmax vs sigmoid" theory was WRONG (both use softmax)

=== NEXT FIX ORDER ===
1. infer_text pipeline (full text generation)
2. Chat template
3. KV cache 256k expansion
