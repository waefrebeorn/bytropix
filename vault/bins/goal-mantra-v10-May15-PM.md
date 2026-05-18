=== bytropix — GOAL PASTE (May 18 — Phase 2 Complete) ===

STATE: gen_text working. Cos-sim 0.9968 vs llama.cpp (quantization noise).
Decode 0.6 tok/s (2× from MoE OpenMP + buffer reuse). Prefill 1.0-1.4 tok/s.
DA v10: 8/10 gaps closed. Remaining: chat template (Gap 7).

=== COMPLETED ===
- GQA Q/gate interleave FIXED: cos-sim -0.51 → 0.9968
- IMRoPE implemented (sections [11,11,10,0], theta=10M) ✓
- MoE quantized path wired (IQ2_XXS/IQ3_XXS/IQ4_XS via blob) ✓
- ref_dumper tool (links libllama.so) ✓
- gen_text pipeline working ✓ (coherent 32-token gen)
- Performance: 0.3→0.6 tok/s (2×) — MoE OpenMP, embedding fix, buffer reuse
- Malloc reduction: 160→5 per forward (pre-allocated buffers)
- Vault papers read: Qwen3.6 arch, Unsloth UD quant, DA v10, 23 MD files
- All mind palace, README, STATUS files updated to Phase 2 status
- MADE_AGENTICALLY_BY_HERMES.md essay written with DA verification
- status-may18-2026.svg diagram generated

=== PENDING ===
P0 — Chat template (gen_text improvement)
P0 — Multi-token cos-sim verification (T>2)
P1 — KV cache for GQA decode (~10% speedup)
P1 — SIMD vec_dot for cos-sim → 1.0
P2 — GPU decode path (~5-10× speedup)

=== GROUND TRUTH ===
Reference: ~/llama.cpp/src/models/qwen35moe.cpp
Dumper: ~/bytropix/ref_dumper (links libllama.so)
Model: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
Hidden dump: DUMP_LAYER_DIR=/tmp/dump_layers

BUILD: make gen_text | MODEL: GGUF
HW: AMD Ryzen 7950X 16C/32T, 64GB DDR5

UNIT TEST:
make test_full_moe && PROFILE=1 ./test_full_moe
# Expect: cos-sim 0.9968, MoE 15ms, SSM 13ms
