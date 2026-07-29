# wubuwizard — 100 Optimizations for Agnostic Model Loading & Hardware-Agnostic Inference

**Goal:** enable the masses to load *all* models into our AGI — agnostic weight loading,
best-in-class handling on *every* hardware class (CPU-only, integrated GPU, consumer dGPU,
datacenter GPU, Apple Silicon, ARM, RISC-V), without vendor lock-in.

Every item cites the authoritative source it was derived from (llama.cpp, vLLM, FLA,
exllamav2, MLC-LLM/TVM, NVIDIA, Intel, peer-reviewed papers). Items already implemented in
wubuwizard are marked `[DONE]`; the rest are the backlog for the perpetual gap-closer loop.

The unifying principle (hardware-agnostic): **dimension-driven loading + a compile-time
invariant core (SSM_D_STATE=128, SSM_K_HEADS=16, DT_RANK=32, CONV_KERNEL=4, KEY_DIM=2048)
with only *varying* dims as runtime globals**, plus a backend abstraction that drops to the
best available kernel per device. No god-headers, no vendor shortcuts, C11 only.

---

## A. Loading & Memory (get the model resident, zero-copy)  [1–20]

1. [DONE] **Safetensors directory + single-file detection** — `wubu_model_init_auto` accepts
   multi-shard checkpoints (glob `model-*-of-*.safetensors`) and single adapter files.
   (HuggingFace safetensors spec; wubuwizard `wubu_model_safetensors_bridge.c`.)
2. [DONE] **LoRA/BTL-3 adapter overlay** — adapter `.safetensors` (peft r=32/α64) applied over
   a base checkpoint via `BTL_BASE` env. (peft; wubuwizard LoRA branch.)
3. **Memory-mapped weights (`mmap`) for zero-copy load** — map GGUF/safetensors directly into
   the address space; weights are paged in on first touch, no bulk `memcpy`. Unified-memory
   (UMA) GPUs get true zero-copy. (llama.cpp GGUF mmap; HF `mmap` on safetensors.)
4. **`--no-mmap` fallback** — for network/disk where mmap is slow (DGX Spark: 56 s w/o mmap),
   read+dequantize directly. (llama.cpp `--no-mmap`.)
5. **GGUF container format** — adopt GGUF (header + tensor data + metadata + tokenizer) for
   portable, versioned, mmap-friendly distribution alongside safetensors. (ggml-org/GGUF.)
6. **Quantized-on-disk, dequantize-on-load** — store Q4_K_M/Q8_0 weights; dequant to F16/F32
   (or keep packed for MMQ) only when GPU needs it. (llama.cpp quantize.)
7. **Lazy layer allocation** — only materialize layers that fit in the target device's VRAM;
   overflow layers stay on CPU. (llama.cpp `llama-fit-params` projected-memory fit.)
8. **Unaligned-block GEMM** — keep Q8_K/Q4_K super-blocks (256-weight) for metadata efficiency.
   (K-quant PR #1684.)
9. **Double-quantized scales** — quantize the per-group scales again (Q8_0 scales for Q4_K),
   cutting metadata ~6→0.5 bits/weight overhead. (K-quant.)
10. **Imatrix (importance-matrix) calibration** — produce I-quants (IQ4_XS, IQ3_M) from a
    diverse calibration corpus (C4 + code) for better quality/byte. (llama.cpp imatrix;
    discussion #5263.)
11. **Per-tensor vs per-channel scales** — pick the scheme that best matches each weight's
    outlier distribution (asymmetric for most, symmetric for Q3_K/Q5_K). (Kaitchup.)
12. **Dynamic per-layer bit allocation** (Unsloth Dynamic 2.0 / `q4_k_xl`) — choose the best
    quant *per layer* from a calibration run, not one global bpw. (Unsloth dynamic GGUF.)
13. **Checkpoint sharding resumable download** — `hf_hub_download` per shard, `.incomplete`
    resume, persistent `/home/wubu/models/<Name>/` (already used; 80 GB survived reboot-proof).
14. **Read-token accelerated HF fetch** — authenticated `HF_TOKEN` raises the anonymous
    rate limit (already applied: Qwen3.6-27B 15 shards pulled in one session).
15. **Model registry dataset (HF)** — `WaefreBeorn/wubuwizard-colonel-registry` inventories
    every model's dims/quants/tokenizer so the loader is config-driven, not hard-coded.
16. **Tokenizer portability** — ship `tokenizer.json` (HF fast tokenizer) + BPE merges +
    chat template; agnostic across all 4 Colonels. (already wired in gauntlet.)
17. **`bf16`/fp16 native weights** — keep bf16 where the hardware has bf16 paths (AMX,
    tensor cores) to avoid f32 upcast cost.
18. **Weight permutation to inference layout at load** — reorder Q/K/V/O/up/gate/down into
    contiguous blocks so the GEMM sees the optimal stride (no gather at inference).
19. **8-bit (GGML_TYPE_Q8_0) intermediate** — store dequantized weights in Q8_0 for the
    CPU int8 path instead of f32 to halve bandwidth.
20. **Page-aligned `mmap` + `madvise(MADV_WILLNEED/HUGEPAGE)`** — prefetch + THP for the
    weight region to kill first-touch stalls.

---

## B. Quantization (fit more model in less memory, less quality loss)  [21–40]

21. [DONE] **Q8_0 path** — near-lossless (≈+0.0004 ppl @7B), default high-quality wubuwizard.
22. [DONE] **F16 path** — full-precision reference path.
23. **Q4_K_M default** — best balance (≈+0.05 ppl @7B, ~4.5 bpw), the community default
    for ≤16 GB VRAM. (llama.cpp + PromptQuorum 2026.)
24. **Q5_K_M high-quality** — near-imperceptible degradation, for 16+ GB VRAM.
25. **Q6_K "almost lossless"** — when you want savings but ≤1% loss; Intel recommends as the
    best all-around GGUF.
26. **IQ4_XS / IQ3_M I-quants** — best quality-per-byte when a good imatrix exists; lets
    larger models fit. (Kaitchup.)
27. **MXFP4 / NVFP4 (microscaling)** — OCP open 4-bit float (block-shared exponent, 32/block);
    Blackwell/MI355 native; ~35 GB for 70B, tiny quality gap with MR-GPTQ. (NVIDIA NVFP4 blog;
    MXFP4 spec; ICLR'26 MR-GPTQ.)
28. **INT4 weight-only** — custom CPU runtime hits 1% of fp32 at 4-bit, 5× over fp32 naive.
    (Shen et al. / Intel Extension for Transformers.)
29. **SmoothQuant** — shift activation outliers into weights so INT8 activation quant works.
    (Xiao et al.)
30. **AWQ** — activation-aware weight quantization preserving salient weights; export target.
31. **GPTQ** — 4-bit post-training quant with OBQ-style Hessian; mature, widely supported.
32. **EXL2** — finest-grained bit control (2–8 bpw mixture); best quality/bit on GPU.
    (exllamav2.)
33. **Grouped-GEMM quant** — pack MoE expert weights for grouped GEMM (see §F).
34. **Per-channel asymmetric quant for embeddings/output** — these layers are outlier-heavy;
    keep them higher precision (Q6_K/Q8_0) in mixed-quant recipes.
35. **KV-cache quant to FP8** — halves KV memory, ~no accuracy loss on H100/H200. (vLLM
    `--kv-cache-dtype fp8`; skip sliding-window layers.)
36. **KV-cache quant to INT8/INT4** — CPU/AMD path (vLLM INT8 KV pending; sglang supports).
37. **K-quant mixed precision inside a layer** — attention values + output proj get 5–6 bit,
    FFN gets 4 bit (Q4_K_M "M" semantics).
38. **Outlier extraction / per-token scales** — isolate the ~0.1% massive outliers to fp16,
    quantize the rest hard (LiteLLM / AWQ-style).
39. **Calibration-set quality gating** — measure KL-div (KLD_99) of candidate quants against
    fp16 logits; auto-reject quants that regress outliers. (llama.cpp #5263.)
40. **Quantization-aware re-pack for the target kernel** — produce the exact byte layout the
    backend's dequant expects (Q8_K super-block, NVFP4 16x16 blocks) so no runtime reshuffle.

---

## C. CPU Kernels (the masses run on CPU/RAM)  [41–58]

41. [DONE] **AVX2 matmul** — wubuwizard builds with `-mavx2 -mfma`. (justine.lol/matmul.)
42. **AVX-512 + VNNI int8** — one-thread-per-physical-core, int8 VNNI dot-product for Q8_K.
    (Intel Xeon study; llama.cpp AVX512.)
43. **AMX int8/bf16** — tile MMA on Sapphire Rapids+; biggest CPU speedup for INT8 quants.
    (Intel AMX; sglang Intel impl 85% mem-eff tensor parallel.)
44. **NUMA-aware split** — bind process to the socket holding the model's RAM (`numactl`);
    for >1 socket use multi-NUMA tensor parallel. (llama.cpp #12303; sglang.)
45. **Thread pool = physical cores** — 1 thread/core; hyperthreads add ~10% only, sometimes
    hurt; cap at 8–16 for small models. (Malakhov CPU study.)
46. **One-thread-per-core pinning + affinity** — `sched_setaffinity` to avoid migration.
47. **Parallel instance per core-group** — for small prompts, N workers on disjoint cores
    beat one worker on all cores. (Malakhov.)
48. **Blocked/tiled GEMM with cache blocking** — tile to L1/L2/L3, avoid cache thrash.
    (justine.lol matmul 233 GFLOPs.)
49. **INT4 decode via LUT** — store 4-bit weights, dequant through a 16-entry LUT (AVX2/512
    as LUT registers) — NoMAD-Attention trick for attention scores too. (NoMAD-Attention.)
50. **Half-precision (F16) SIMD path** — `_mm256_cvtph_ps` on AVX2/512; half weights, f32 math.
51. **Prefetch + software pipelining** — overlap dequant with the fma stream.
52. **Repetition/DRY penalties (already in wubuwizard)** — repeat-penalty 1.05/1.1, dry-mult
    0.5/1.2, dry-base 1.75 — slashed agent-loop failure Q8 34%→3%. (your tuning paste.)
53. **Chunked prefill (already exact)** — O(T·d) chunked SSM; 256K verified. (wubuwizard.)
54. **GDN chunkwise-parallel prefill [DONE this session]** — exact WY/UT closed form,
    opt-in `WUBU_GDN_CHUNK`; proven 0.0 diff vs scalar at all C. (veitner/sustcsonglin/GDN
    arXiv:2412.06464 — the GPU-ready matmul form.)
55. **QKNorm (already present)** — L2-norm of q/k for delta-rule stability. (GDN paper.)
56. **Partial-offload CPU stub** — when GPU VRAM < model, run the overflow layers on CPU
    with a zero-copy bridge (see §E).
57. **ARM NEON / SVE path** — port the matmul to NEON (int8 dot) for Apple/Android/Ampere.
    (MLC-LLM ARM; llama.cpp NEON.)
58. **RISC-V V extension path** — RVV int8 GEMM for Banana-Pi/ Vision-Five class SBCs.
    (llama.cpp RISC-V build.)

---

## D. GPU / Accelerator Backends (best on every GPU)  [59–74]

59. **CUDA MMQ (mixed-matmul) kernels** — fused dequant+GEMM for Q8_K/Q4_K on tensor cores.
    (llama.cpp `GGML_CUDA_FORCE_MMQ`.)
60. **cuBLAS path for f16/bf16** — when weights stay fp16, call cuBLAS directly.
61. **Vulkan backend** — single cross-vendor shader for NVIDIA/AMD/Intel GPUs + MoltenVK.
    (MLC-LLM; wubuwizard GPU shim goal.)
62. **Metal backend** — Apple-Silicon-native for the Mac masses. (MLC-LLM / llama.cpp metal.)
63. **ROCm (HIP) backend** — AMD dGPU path. (MLC-LLM ROCm.)
64. **WebGPU / browser** — run in-browser on any GPU via WebGPU (up to 59% faster prefill than
    WebGPU-on-Apple via Vulkan/MoltenVK). (arXiv:2605.20706 "Llamas on the Web".)
65. **OpenCL fallback** — broadest hardware reach (old GPUs, mobile). (MLC-LLM OpenCL.)
66. **NVFP4 tensor-core path** — Blackwell 4–5× over fp8 on the same silicon. (NVIDIA NVFP4.)
67. **FP8 (e4m3) KV + attention** — H100/H200 native. (vLLM fp8-kvcache blog.)
68. **CUDA Graphs** — capture the decode kernel graph, kill per-step launch overhead.
    (Megatron/DeepSeek prod features.)
69. **Flash-Decoding (split-KV)** — parallelize attention over KV for long context, up to
    8× faster generation. (Tri Dao; pytorch flash-decoding blog.)
70. **FlashAttention-4 page-size flexibility** — arbitrary KV page sizes, up to 2.4× for small
    pages, 4.37× for small query (decode). (modal.com FA4.)
71. **Tensor-parallel split (`--split-mode tensor`)** — split weights *and* KV across GPUs
    via reductions; best token-gen on NVLink. (llama.cpp multi-gpu.md.)
72. **Pipeline (layer) split (`--split-mode layer`)** — default; each GPU holds a layer slice;
    best prefill on weak interconnect. (llama.cpp.)
73. **Multi-GPU auto-fit (`llama-fit-params`)** — compute projected VRAM, pick n_gpu_layers +
    tensor-split + per-tensor overrides automatically. (llama.cpp #18049.)
74. **GPU→CPU partial offload bridge** — keep hot layers on GPU, cold on CPU, with async
    copy (see §E).

---

## E. Heterogeneous / Offload & Paging (models bigger than any one device)  [75–84]

75. **Layer pipeline offload** — contiguous layer slices across GPU+CPU+NVMe; compute as
    layers stream in. (llama.cpp n_gpu_layers; PIE memory pooling.)
76. **Tensor-split offload** — per-tensor `-ot` overrides route specific experts/layers to
    CPU/GPU. (llama.cpp #18049 overrides.)
77. **NVMe swap (CPU RAM + disk)** — host LLMs larger than RAM by paging weights from NVMe.
    (PIE; Malakhov §2.1.)
78. **Expert offload for MoE** — keep routed experts on disk, page the active ones per token
    (see §F). (AWS Neuron All-Experts algorithm.)
79. **UMA zero-copy (Apple/APU/IGPU)** — weights live in unified memory, no copy GPU↔CPU.
    (llama.cpp UMA issue #21827.)
80. **KV-cache paging (PagedAttention)** — allocate KV in non-contiguous pages, ~0 waste,
    enables big contexts on fixed VRAM. (vLLM arXiv:2309.06180.)
81. **Prefix caching** — reuse KV blocks for shared system prompts; +250% throughput on
    Qwen3-32B, up to 90% cost cut. (vLLM prefix caching; reddit test.)
82. **Chunked prefill** — split long prompts into chunks interleaved with decode to keep
    latency flat under continuous batching. (Modular handbook.)
83. **CPU+GPU hybrid attention** — attention may stay on CPU for huge ctx while GPU does FFN
    (DocShotgun NUMA note) — keep the math identical (wubuwizard does this via FORCE_CPU_SSM_SEQ).
84. **Async weight staging** — overlap next-layer DMA with current-layer compute.

---

## F. MoE Handling (KAT-Coder, DeepSeek-class)  [85–92]

85. [DONE] **MoE config detection** — wubuwizard reads `qwen3_5_moe_text` (KAT: 256 experts/8
    active, shared expert). (config.json.)
86. **Grouped GEMM** — batch all active experts' GEMMs into one kernel for high utilization.
    (Megatron `--moe-grouped-gemm`; DeepSeek-V3 prod.)
87. **Expert parallelism** — distribute experts across GPUs/devices. (vLLM `--enable-expert-parallel`.)
88. **Shared-expert overlap** — fuse shared-expert compute with the routed dispatch.
    (Megatron `--moe-shared-expert-overlap`.)
89. **Group-limited routing (DeepSeek-V3 style)** — `n_group`/`topk_group` for balanced load.
    (AWS Neuron.)
90. **All-to-All token dispatch** — efficient routed-expert communication. (Megatron
    `--moe-token-dispatcher-type alltoall`.)
91. **Per-expert quant** — quantize each expert independently (some experts tolerate lower bpw).
92. **Dropless MoE + sync-free exec** — CUDA Graphs + sync-free for steady decode. (Megatron.)

---

## G. Decoding & Serving (throughput for the AGI swarm)  [93–100]

93. **Continuous batching** — iteration-level scheduling; finished seqs free slots immediately;
    up to 23× throughput, flatter p50 latency. (Anyscale; Modular handbook.)
94. **Speculative decoding (EAGLE-2 / Medusa / Lookahead)** — draft tree + tree-attention
    parallel verify, lossless; big decode speedup on GPU. (arXiv:2406.16858; SpecInfer.)
95. **Medusa heads** — train MLPs to predict multiple tokens from LLM features (no draft model).
    (Cai et al. 2024.)
96. **Lookahead / n-gram draft** — cheap CPU-friendly draft from the model's own n-grams.
97. **Repetition + DRY already tuned** — see #52; essential for long agent rollouts.
98. **Sampling config defaults** — temp 0.6 / top_p 0.95 / top_k 20 (your paste); ctx 131072,
    ub 2048 — already in wubuwizard `gen_text`.
99. **Attention backend select** — FlashAttention / FlashDecoding / PagedAttention chosen per
    device + query length automatically (long ctx → split-KV; short → fused).
100. **Backend auto-dispatch (GGML-style)** — `ggml_backend_*_supports_op` lets each device
     accept/reject ops and fall back to CPU; one graph runs on any mix of devices. (GGML
     backend API; FOSDEM'25.) This is the *keystone* of hardware-agnostic loading: a single
     compute graph, N backends, zero vendor lock-in.

---

## Implementation status (this session)
- **[DONE]** A1 safetensors dir + adapter loading; A2 LoRA; A21 Q8_0/F16; C41 AVX2; C53 chunked;
  C54 **GDN chunkwise-parallel (research-backed, 0.0 diff)**; C55 QKNorm; F85 MoE detect;
  G97/G98 repetition+DRY + sampling defaults.
- **Backlog (priority order for the gap-closer loop):** A3 mmap zero-copy → B23 Q4_K_M →
  D59 CUDA MMQ → E75 layer offload → G93 continuous batching → D61 Vulkan → A10 imatrix →
  B27 MXFP4 → F86 grouped GEMM → G94 EAGLE speculative.

## Research basis (authoritative)
llama.cpp (GGUF, K-quant #1684, imatrix #5263, multi-gpu, fit-params #18049, matmul justine.lol);
vLLM (PagedAttention 2309.06180, prefix caching, fp8-kvcache); Flash Linear Attention (GDN
arXiv:2412.06464, GLA 2312.06635, chunkwise parallelism); veitner "Chunkwise Gated Delta Rule";
sustcsonglin "DeltaNet Explained II"; exllamav2 (EXL2); MLC-LLM/TVM (Vulkan/Metal/ROCm/WebGPU,
arXiv:2605.20706); NVIDIA NVFP4 + MXFP4/MR-GPTQ (ICLR'26); Intel AMX/Xeon study; Megatron-Core
MoE (grouped GEMM, expert parallel); Anyscale continuous batching; Tri Dao Flash-Decoding;
Malakhov CPU-only deployment; SmoothQuant; AWQ/GPTQ.

*This catalog is the AGI's loading/handling bible: agnostic weights in, best path per device out.*
