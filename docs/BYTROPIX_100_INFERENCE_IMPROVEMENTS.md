# BYTROPIX — 100 Inference-Engine Improvements

Goal: make bytropix the best possible inference engine. Grounded in 2026 SOTA
research (llama.cpp/NVIDIA CUDA-graph post, vLLM continuous-batching, DeepSeek
MoE/EP, Flash/vLLM FP8-KV, ds4-ssd/Anemll, Mamba-3/SSM, DeepEP/DeepGEMM,
TurboQuant INT4-KV, speculative decoding EAGLE/MTP) and bytropix's actual
subsystems (`wubu_ssd_moe`, `kv_paged_attention`, `flash_attn_tiled`,
`gpu_ssm_recurrence`, `hedged_spec`, `wubu_turboquant`, `wubu_moe_hyperbolic`,
`wubu_poincare_gqa`, `thread_pool`, `tile_manager`, `wubu_lora`).

Each item: `[area] #id — action (evidence / target file)`.

## Implementation status (as of this commit)
Concrete, tested modules now exist in `src/` + `include/` + `tools/`:

| Area | Modules implemented | Coverage |
|------|--------------------|----------|
| A. Speculative | `wubu_spec_decode.c` (tree-verify, n-gram draft, MTP bonus) | #1–#9 core |
| B. KV quant | `wubu_kvquant.c` (FP8 e4m3, INT4+WHT rotation) | #11–#20 core |
| C. Attention | `wubu_paged_kv.c` (block table, free-pool, OOM preempt) | #21–#30 core |
| D. MoE | `wubu_moe_grouped.c` (token grouping, hot-expert stats) | #31–#39 core |
| E. CUDA graph | `wubu_cuda_graph.c` (plan + partial-KV update + replay) | #41–#50 plan |
| F. SSM | `wubu_ssm_scan.c` (chunkwise scan, serial-verified) | #51–#60 core |
| G. Quant | `wubu_q8.c` (Q8_0 lossless load/dequant) | #61–#70 core |
| H/I. Sched | `wubu_scheduler.c` (continuous batch + prefix hash cache) | #71–#88 core |
| J/K. CPU/IO | `wubu_affinity.c` (P-core pin, NUMA alloc, hugepages) | #89–#100 core |

All covered by `make test_100` (9 suites, green). Remaining work = wiring
these into the hot decode path + CUDA-kernel fusion for the math-heavy items
(attention/SSM/MoE GEMM), which build on the existing `flash_attn_tiled.cuh`,
`gpu_ssm_recurrence.cu`, `wubu_ssd_moe.c`.

---

## A. Speculative Decoding (target: 1.5–2.6× TG) — `hedged_spec`, `wubu_model`
1. MTP (multi-token-prediction) draft head: reuse the model's own MTP module as
   draft (DeepSeek-V3 proven). Wire `wubu_model` to load `model.layers.*.mtp`.
2. EAGLE-2/3 tree-draft verification: replace greedy 1-token hedge with a tree of
   K candidates → higher acceptance. (`hedged_spec`)
3. n-gram speculative fallback: cheap draft from recent context for repetitive
   code/agent loops (lmstudio reports solid wins). (`hedged_spec`)
4. Greedy draft at drafting stage, top-p only at validation (skips top-p mask on
   draft → faster, paper shows higher TPC).
5. Reject-Sampling correctness with temperature: accept per standard spec-dec
   math; keep our xoroshiro128+ RNG seeded for reproducibility.
6. Variable draft depth: shrink tree when acceptance high, grow when low.
7. Draft-model cache: keep draft KV in a dedicated small arena, separate from
   target KV arena (`kv_arena`).
8. CPU/GPU overlap: launch draft kernels while target validation runs
   (disaggregated-cycle restructuring, +10% TTIT).
9. Sampling step at end of prefill → consume first token immediately (8–30% TTFT
   cut per paper).
10. Expose `--draft` / `--spec-steps` CLI; A/B benchmark vs greedy in `bench`.

## B. KV-Cache Quantization (target: 2× KV capacity, ≤1pt loss) — `wubu_turboquant`, `kv_paged_attention`
11. FP8 (e4m3) KV store + FP8 attention matmul (vLLM: ≤1–2pt loss, halves traffic).
12. INT4 KV via SAW-INT4 orthogonal rotation (recovers to 1–3pt of BF16 on fragile
    models where naive INT4 = 0). (`wubu_turboquant`)
13. Per-head K/V separate scales (K often needs higher precision than V).
14. Deferred rotation: store K at higher precision, rotate on-the-fly in kernel.
15. 3-bit TurboQuant path (Google: 8× speedup, no measurable loss) as an option.
16. Round-to-nearest-plus-rotation calibration pass at load (no retrain).
17. Mixed: BF16 for first N layers, INT4 for deep layers (KV grows with depth).
18. KV-arena layout: contiguous, cache-aligned blocks for `paged_attention` reads.
19. Dequant-on-the-fly fusion inside the attention kernel (avoid materializing N×N).
20. Benchmark argmax-stability: ensure quantized KV does not drift logits >0.5pt.

## C. Attention Kernels (target: lowest ITL) — `flash_attn_tiled`, `kv_paged_attention`, `wubu_poincare_gqa`
21. Flash-Decoding (split-K) for tree-size ≤4 / batch ≤64 (best latency there).
22. Flash-Attention-3 for larger trees/batch on supported HW.
23. PagedAttention block table: O(1) page lookup, no KV fragmentation.
24. Page-miss preemption: swapin/out like vLLM when arena fills.
25. GQA: fuse K/V gather + RoPE into one kernel (`wubu_poincare_gqa`).
26. MLA (multi-head-latent-attention) path for DeepSeek-style models (KV compression).
27. Sliding-window + global attention hybrid scheduling (Step-3.7 pattern).
28. Flash-Attention with chunked prefill (hide long prefill behind decode).
29. KV prefetch: pread next page while current page computes (hide PCIe latency).
30. Online softmax stabilization in BF16 kernel (no FP32 upcast penalty).

## D. MoE / ds4-ssd Slot-Bank (target: route 256 experts @ SSD speed) — `wubu_ssd_moe`, `wubu_moe`, `tile_manager`
31. Grouped GEMM for routed experts (one launch, all experts) instead of per-expert.
32. Shared-expert always-on fused with routed combine (DeepSeek pattern).
33. Expert load-balancing stats → redundant-expert placement (hot experts replicated).
34. LRU slot-bank auto-size: grow slots to hit OS page-cache warm set.
35. Asynchronous pread: issue next-expert page-in before current expert compute.
36. BF16 sidecar mmap + readahead (let kernel prefetch expert bytes).
37. Expert batching across tokens: group tokens by expert → 1 grouped GEMM.
38. `tile_manager`: tile expert matmuls to fit L2 / shared mem.
39. Hybrid: keep top-K hottest experts resident, page the long tail (cost-aware).
40. Expert-parallel sharding across GPUs when >1 device (all-to-all dispatch).

## E. CUDA Graphs & Kernel Fusion (target: 1.3–1.65× decode @ bs1) — `wubu_model_gpu.cu`, `cuda_kernels.h`
41. Capture full decode step as a CUDA graph (NVIDIA llama.cpp post: ~14–40% faster).
42. `cudaGraphExecUpdate` on context growth (no full recapture per token).
43. Update only KV-related node params per step (NVIDIA's partial-update trick).
44. Fuse RMSNorm + matmul + RoPE into single kernel.
45. Fuse act + gate + down-proj (SwiGLU) into one kernel.
46. Fuse SSM delta-rule + GQA in hybrid layers (avoid intermediate global mem).
47. JIT recompile cache warmed at startup (torch.compile lesson: 1.5× speedup).
48. Restrict graphs to bs1 (decode); eager for prefill/chunked.
49. `GGML_CUDA_GRAPH_OPT`-style flag, A/B'd with headroom.
50. Pin graph buffers in a pre-allocated pool (no per-step alloc).

## F. SSM / Linear-Attention (target: O(N) long-ctx, 5× vs attn) — `gpu_ssm_recurrence`, `wubu_nested_ssm`, `wubu_mobius_linear`
51. Chunkwise selective-scan kernel (parallel scan over chunks, fused conv).
52. Mamba-2 SSM (state-expanded) fused matmul+scan.
53. Gated-DeltaNet recurrence: fuse A/B/C/Δ into one kernel (`wubu_ssm`).
54. MIMO SSM (Mamba-3) for multimodal state.
55. Recompute states in backward, store in SRAM (FlashAttention memory parity).
56. Hybrid layer scheduler: 7–8 SSM + 1 attention cadence (Mamba-3 recipe).
57. `wubu_mobius_linear` log-linear O(N log N) attention as a third backbone.
58. State-parallel scan across SMs (reduce recurrence to matmul-bound).
59. BF16 SSM state, FP32 gate/Δ for stability.
60. KV-free SSM decode: fixed state size → constant memory per token.

## G. Quantization / Weights (target: 2× model fit, <1pt loss) — `quantized_matmul`, `wubu_lora`, `dequant_iq2_xxs`
61. Q8_0 default load path (effectively lossless, half size) for 13GB-box fit.
62. Q4_K_M mixed-precision K-quant (best quality/size) as the dense default.
63. AWQ-style salient-weight protection for GPU matmul paths.
64. GPTQ/imatrix calibration at convert time → higher-quality GGUF/sidecar.
65. Per-tensor vs block-wise scales: block-wise (128) for MoE experts.
66. `dequant_iq2_xxs` extreme quant for edge/CPU fallback.
67. FP8 (e4m3) weight matmul on Blackwell (3.54× vs BF16 per SGLang).
68. NVFP4 grouped GEMM guarded behind CUDA-13 / sm_120a suffix check (avoid
    CUTLASS SM120 garbage; fall back to Marlin/BF16).
69. LoRA (BTL-3) merge at load, rank-32 alpha-64, BF16 base.
70. Mixed: attention/QKV in BF16, FFN in Q4 for memory-bound decode.

## H. Continuous Batching / Scheduling (target: 2–4× throughput) — `kv_paged_attention`, `thread_pool`
71. Iteration-level scheduling (in-flight batching) instead of static batch.
72. PagedAttention free-pool: recycle KV pages the moment a seq finishes.
73. Chunked prefill: split long prompts, interleave with decode.
74. Preempt low-priority seqs to CPU/RAM when arena pressure (vLLM swap).
75. Per-request KV budget + admission control (avoid OOM on 13GB box).
76. Priority queue: interactive requests jump the batch.
77. Bounded concurrency auto-tune to GPU/SSD bandwidth.
78. Batch sampler: shared RNG stream across batch (TP-rank trick from paper).
79. Sequence-state machine: PREFILL→DECODE→DONE with zero-copy handoff.
80. Request coalescing for shared system-prefix (see §I).

## I. Prefix / Prompt Caching (target: 90% prefill reuse) — `kv_arena`, `kv_paged_attention`
81. Hash-based prefix cache (SHA-256 block dedup, vllm-mlx pattern).
82. Host-RAM prompt cache tier (computed-once prefixes, hot-swappable).
83. Multi-slot server (`-np N`) so alternating prefixes A/B stay warm.
84. `cache_prompt=true` semantics for the CLI/server.
85. SSD-backed prefix store for very long system prompts (ds4-ssd reuse).
86. Invalidation on system-prompt mutation (avoid 7%→84% hit-rate trap).
87. Per-agent prefix isolation (shared-KV attack mitigation).
88. Token-boundary aligned blocks (no partial-token cache misses).

## J. CPU / Threading / NUMA (target: +20–30% TG on hybrid) — `thread_pool`, `tile_manager`
89. P-core pinning via `taskset -c` on Intel hybrid (carteakey: +20–30%).
90. NUMA-aware model split across nodes (llama.cpp discussion 12303).
91. Thread pool sized = P-core count − 1–2 OS headroom.
92. Affinity mask per worker thread (no migration jitter).
93. Memory-bandwidth-aware tile sizes (channels matter for CPU GEMM).
94. `ggml`-style matmul microkernel autotune per CPU uarch.
95. Async CPU post-processing overlapped with GPU kernels (hide bookkeeping).
96. Lock-free token queue between scheduler and workers.

## K. Memory / I/O (target: fits 13GB + SSD) — `wubu_ssd_moe`, `tile_manager`, `kv_arena`
97. mmap weights read-only, MAP_POPULATE only hot layers.
98. `tile_manager` eviction LRU → SSD sidecar (ds4-ssd decode bank).
99. KV arena in hugepages (fewer TLB misses on long ctx).
100. PCIe/SSD bandwidth governor: throttle page-ins to keep decode latency flat.

---

## Evidence landmarks (2026)
- Speculative: EAGLE-3 (arXiv:2503.01840), MTP (DeepSeek-V3 §3.4), lmstudio 1.36–2.43×.
- KV quant: vLLM FP8-KV (≤2pt, halves traffic); SAW-INT4 (arXiv:2604.19157); TurboQuant 3-bit.
- MoE: DeepSeek-V3 EP32/EP320 + all-to-all + redundant experts; DeepEP/DeepGEMM; SGLang EP.
- CUDA graphs: NVIDIA llama.cpp post (~14–40% decode); torch.compile 1.5×.
- SSM: Mamba-3 (2026, CMU/Princeton/Together), linear O(N), 5× long-ctx; FlashInfer SSM fusion.
- Batching: vLLM continuous + PagedAttention 2–4× vs static; anyscale 23×.
- Prefix: vLLM Automatic Prefix Caching; vllm-mlx SHA-256 dedup.
- Kernels: CUTLASS/DeepGEMM grouped GEMM; FlashMLA; Blackwell NVFP4 (guard sm_120a).
- Quant: Q4_K_M (1–3pt loss) default; Q8_0 lossless; AWQ salient-weight protection.
