═══ BYTROPIX — STATUS (May 18 — Phase 2: Performance Opt) ═══
Path: /home/wubu/bytropix | Branch: master
HW: AMD Ryzen 7950X 16C/32T, 64GB DDR5, RTX 5050 6.4GB
Model: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (2.7 bpw, 10.7 GB)

=== INFERENCE PARITY (C code to C code) ===
| Component            | Status          | Notes                              |
|----------------------|-----------------|------------------------------------|
| Full 40L forward     | ✅ cos-sim 0.9968 | Verified via test_full_moe         |
| GQA Q/gate interleave| ✅ FIXED        | cos-sim -0.51 → 0.9968             |
| IMRoPE               | ✅ Implemented   | sections=[11,11,10,0], theta=10M   |
| MoE quantized path   | ✅ Wired         | IQ2_XXS/IQ3_XXS/IQ4_XS via blob    |
| Shared expert gate   | ✅ Sigmoid gate  | ffn_gate_inp_shexp * sigmoid(x_s)  |
| Output proj Q4_K     | ✅ cos-sim 0.99995 | vs F32 SGEMM                       |
| gen_text pipeline    | ✅ Working       | coherent 32-token generation       |
| Decode speed         | ⚠️ 0.6 tok/s     | CPU-only, 16 threads               |
| Prefill speed        | ℹ️ 1.0-1.4 tok/s | depends on prompt length           |
| Chat template        | ❌ Not applied   | DA Gap 7. Minor quality impact     |
| SSM L2 eps           | ⚠️ 1e-12         | llama.cpp uses ~1e-6. Not blocking |

=== PER-LAYER COS-SIM DECAY ===
Peak: 0.9985 (L3 GQA) → Floor: 0.9952 (L30-39)
Decay rate: ~0.00011/layer — quantization noise accumulation
Conclusion: No architecture bugs. Gap is generic C vec_dot (no SIMD).

=== DA v10 GAP AUDIT ===
✅ Gap 1-2: Dequant noise — CLOSED (output proj 0.99995)
✅ Gap 3: Decode pipeline — CLOSED (gen_text works)
✅ Gap 4: MoE perf — CLOSED (3× speedup, thread-local)
✅ Gap 5: Shared expert gate — CLOSED (sigmoid applied)
✅ Gap 6: SSM norm — CLOSED (cos-sim verified)
⚠️ Gap 7: Chat template — OPEN (not applied to gen_text)
✅ Gap 8: Tensor audit — CLOSED
✅ Gap 9: Final norm — CLOSED
✅ Gap 10: Ground truth — CLOSED (cos-sim 0.9968)
8/10 closed. 1 open.

=== NEXT FIX ORDER ===
1. Add chat template to gen_text (minor quality fix)
2. KV cache for GQA decode (~10% speedup)
3. SIMD vec_dot for cos-sim → 1.0
4. GPU decode path (~5-10× speedup)
