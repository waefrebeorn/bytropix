# 006 — Arena allocator + SoA layout (game-console data-oriented design)

Source: "Data-Oriented Design" (dataorienteddesign.com/dodbook.pdf); Wikipedia
"Data-oriented design"; Ryan Fleury "Untangling Lifetimes: The Arena Allocator";
gameprogrammingpatterns.com/data-locality.html; classmethod SoA 3× faster.

## Core idea
Console games hit the same wall we hit: the CPU cache, not FLOPS, is the
bottleneck. Two discipline transfers:
1. **Arena allocator**: one `malloc` per request (or per engine step), free all at
   once. Eliminates per-token malloc/free churn in the decode loop — our current
   `quantized_matmul` allocates `w32` buffers per call.
2. **Structure-of-Arrays**: store activations/states as parallel `float[]` arrays,
   not `struct{float x,y,z}[]`. Our KV cache is already SoA (element-indexed
   float array) — extend the discipline to activation buffers and per-layer temps.

## Triple-DA
- P1 correctness: arena just changes *lifetime management*, not values. SoA changes
  *layout*; accessors must match. Pure refactor, no numeric change. ✓
- P2 privacy: pure C. ✓
- P3 robustness: arena must not overflow mid-request (abort-with-diagnostic, not
  silent corruption). Per-thread scratch arenas (Fleury) avoid contention.

## Implementation plan
- `wubu_arena.c/.h`: linear allocator `arena_alloc(bytes, align)`, `arena_temp_begin
  /end` for scratch, `arena_reset` per step. Used by `quantized_matmul` for its
  `w32` temp (no more per-call malloc/free).
- Audit every `malloc` in the decode hot path; replace with arena or stack scratch.

## Test oracle
- Stress: run 1000 decode steps; assert peak malloc count in hot path ≈ 0 (all
  arena). ASAN-clean.
- Correctness: same forward output (cosine 1.0) with arena vs malloc path.
