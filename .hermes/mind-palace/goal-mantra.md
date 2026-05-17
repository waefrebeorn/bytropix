═══ GOAL PASTE (May 17 v5 — TGT/manifold cleared) ═══
PROJECT: bytropix — Custom Qwen3.6-35B-A3B inference engine
MODEL: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (7-type mixed quant)

=== STATUS ===
Both RoPE bugs: FIXED
SSM delta net math: VERIFIED (identical to llama.cpp)
TGT/manifold: AUDITED (clean — not in inference path)
ssm_a values: CHECKED (all negative, correct for DeltaNet)
Weight layouts: VERIFIED (all 14 components match)
Hidden state: STILL ORTHOGONAL (cos-sim 0.0167 vs ref)

=== NEXT: LAYER-BY-LAYER DUMP ===
The bug is at a level too low for code reading.
Must dump hidden state after each layer from BOTH engines.
Use llama.cpp LLAMA_DUMP_LAYER_DIR + matching dumps in our engine.
Find first divergent layer. Then dump all intermediates for that layer.

=== BUILD ===
rm -f src/cuda_kernels.o infer_text && make infer_text
