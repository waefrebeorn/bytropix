# Plan — May 18, 2026 — PHASES 0-4 DONE. PHASE 5+6 INFRA BUILT.

## Phase 0: CORE INFERENCE ✓
- GQA Q/gate interleave fixed (cos-sim -0.51 → 0.9968)
- All 40 layers verified: cos-sim > 0.995
- MoE quantized path wired (IQ2_XXS/IQ3_XXS/IQ4_XS via blob)
- Shared expert quantized path (Q5_K/Q6_K)
- Per-layer dump via ref_dumper

## Phase 1: DA v10 Closure ✓
- All 10 gaps closed (cos-sim 0.9969, chat template fixed)

## Phase 2: Performance Optimization ✓
- MoE OpenMP: 3× speedup (44ms→15ms per layer)
- Embedding file: opened once, closed at end
- Decode: 0.3→0.7 tok/s (2.3×)

## Phase 3: SIMD vec_dot ✓
- Q4_K/Q5_K: SSSE3 _mm_maddubs_epi16
- Q6_K: SSE4.1 _mm_cvtepi8_epi16
- Cos-sim: 0.9968 → 0.9970

## Phase 4: KV Cache ✓
- K/V cache per GQA layer [10][4096][512]
- Append-only: decode attends to ALL cached positions
- cos-sim T=1 identity preserved

## Phase 5: Island Boy Batch Decode [INFRA BUILT — NEXT]
### Done ✓: All quant types supported
- Q2_K (84B/256), Q3_K (110B/256) — dequant-to-F32 fallback
- Q8_0 (34B/32) — on-the-fly dequant SGEMM
- IQ2_S (88B/256) — dequant-to-F32 fallback
- BF16 (2B/elem) — on-the-fly SGEMM
- save_last_hidden field for h_39 capture
### TODO:
- SSM state save/restore buffers for verify rollback
- Batch-forward draft tokens (B=4) through 40 layers
- Prefetch weights: _mm_prefetch for next layer

## Phase 6: MTP Speculative Decode [INFRA BUILT — NEXT]
### Done ✓: MTP model loading
- wubu_mtp_load loads blk.40 + nextn from same GGUF
- wubu_mtp_draft_forward produces draft tokens (greedy)
- mep_head_t with KV cache for blk.40 GQA
### TODO:
- Draft: generate 3-4 candidate tokens via blk.40
- Verify: batch-forward all candidates through 40 layers
- Acceptance: compare argmax, accept longest prefix
- SSM state rollback on partial accept
- KV cache rollback

## Phase 7: Hardware Saturation [FUTURE]
- AVX2/AVX-512 vec_dot paths
- NUMA-aware thread scheduling
- Quantized scatter/gather for IQ MoE

## Target Performance
| Phase | tok/s | vs baseline |
|-------|-------|-------------|
| Current (P4) | 0.7 | 1× |
| Phase 5 (batch B=4) | 1.2-1.5 | 1.7-2.1× |
| Phase 6 (MTP spec) | 2-4 | 2.7-5.7× |
| Phase 7 (HW sat) | 3-5 | 4-7× |
