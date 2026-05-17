═══ GOAL PASTE (May 17 v17 — MoE verified cos-sim 1.0, stale-binaries corrected) ═══
PROJECT: bytropix — Custom Qwen3.6-35B-A3B inference engine
MODEL: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
┌─────────────────────────────────────────────────────────────┐
│ ARCH: qwen35moe │ 40L │ D_MODEL=2048 │ Q16/KV2 │ SSM16/32 │
│ MoE: 256 experts │ top-8 │ shared │ IQ2_XXS gate/up/down │
└─────────────────────────────────────────────────────────────☆

=== VERIFIED (no bugs found) ===
1. lazy_moe_decode vs wubu_moe_forward: cos-sim 1.000000 (same input, fresh build)
2. Per-expert dequant: bit-identical (gguf_read_tensor_f32 vs dequant_multi_expert_contiguous)
3. Routing: identical top-8 expert selection
4. Shared expert: identical
5. Per-expert forward: identical access patterns

=== STALE DATA RETRACTION ===
- Previous "MoE=1 cos-sim 0.337" = stale binaries (source newer than compiled infer_text)
- Previous "lazy_vs_lib cos-sim 0.612" = stale binary comparison
- Previous "layer 0 cos-sim 0.928" = compared MoE output vs residual dump (wrong file)
- All three were artifacts of comparison methodology, not real bugs

=== TOOLS ===
- infer_text: main inference (NOGPU=1 for CPU, MOE=1)
- check_topk_agreement: verifies routing agreement between lazy/lib paths
- check_dequant_agreement: verifies dequant bit-identity

=== BUILD ===
make infer_text
NOGPU=1 MOE=1 ./infer_text /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "Hello" 1
