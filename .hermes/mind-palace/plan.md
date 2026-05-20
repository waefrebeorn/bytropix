# Plan — Phase 25+ Roadmap (May 20 PM)

## Done (Phase 24)
- Fused Q4_0 K→score decode kernel
- Fused Q4_0 V-weighted sum kernel  
- Batched online softmax kernel
- Sliding window attention (GQA_WINDOW env)

## Done (Phase 25)
- Q5_K fused quant matmul (no bv[256] spill)
- Q6_K fused quant matmul (no bv[256] spill)
- SSM beta/alpha fused decode kernel (replaces cuBLAS + element-wise)
- 28 DeepSeek papers in vault
- External benchmark reference saved
- DA audit v23 — all doc staleness mapped

## Phase 26 — Fuse SSM post-matmul for N=1 decode
- Fuse: conv1d + SiLU + split + L2 norm → 1 kernel (currently 5 launches)
- Reduces ~50μs/launch overhead × 30 layers = ~1.5ms per token
- Post-fusion: verify correctness vs old separate-kernel path

## Phase 27
- Nsight profiling of full decode pipeline
- Identify REAL bottlenecks (stop guessing)
- Target: find why we're 4-7x slower than external reference

## Phase 28
- MoE router on GPU
- Reduce CPU ↔ GPU sync for expert selection

## Phase 29
- Chunked prefill (from Qwen2.5-1M / Qwen3 paper)
- 256k cos-sim verification pipeline

## Ongoing
- Keep docs in sync with code (auto-DA after each commit)
- Clean up repo binaries (make clean, .gitignore)
