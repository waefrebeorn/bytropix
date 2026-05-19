# State — May 19, 2026 (02:15) — LLAMA DEPS KILLED. NV64 RDRAM DESIGN DOC WRITTEN.

## REAL STATUS
No libggml-cpu.so dependency. All vec_dot self-contained in quantized_dot_generic.c. gen_text: 2.1 tok/s decode, 7.8 tok/s prefill. NV64 RDRAM ring buffer design ready for Phase 9.

## Phase 7 Complete
GQA AVX2 + stack buf, AVX2 vec_dot Q4/Q5/Q6, prefetch. Decode 0.7→2.1 tok/s (3×).

## Cleanup
- Dead extern declarations removed from quantized_matmul.c
- libggml-cpu.so refs purged (only ref_dumper* targets still link llama)
- All vec_dot types self-hosted: Q4_K/Q5_K/Q6_K (AVX2+SSE+generic), IQ2_XXS/IQ3_XXS/IQ4_XS (generic)

## Design
- `.hermes/mind-palace/nv64-rdram-ring-buffer.md` — full NV64 RDRAM ring buffer architecture
- 64-slot ring buffer with time-sync token ticks
- CPU/GPU tandem: split at layer 20, overlapped compute
- Prefetch agent graduated T2→T1→T0
- Distributed extension: ring slot = machine[i % N]

## Next
Phase 8: MoE optimization (AVX2 IQ2_XXS/IQ3_XXS vec_dot)
Phase 9: NV64 ring buffer implementation + GPU tandem