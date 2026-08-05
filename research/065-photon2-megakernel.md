# Moondream Photon-2 — Megakernel Inference Compiler

## Source
- Blog: https://moondream.ai/blog/photon-2-launch
- Released: August 3, 2026

## Overview

Photon 2.0 is an inference compiler (not a model) that produces **megakernels** —
single GPU kernels that fuse the entire inference pipeline, eliminating CPU-GPU
synchronization overhead.

### Key Concepts

#### Megakernel
A megakernel is a single large GPU kernel that runs the _entire inference_ pipeline
on the GPU alone, without CPU intervention between operators. This contrasts with
traditional inference engines (vLLM, SGLang) which launch dozens of separate GPU
kernels coordinated by the CPU, creating synchronization overhead.

#### Inference Compiler
Photon 2 is built as a generalized inference compiler that:
- Takes model architectures (Moondream, Qwen, Gemma) as input
- Fuses operator boundaries where optimization opportunities exist
- Produces a single megakernel per model

### Architecture Support
Photon 2.0 launches with support for:
- **Moondream 2 and 3** — vision-language models
- **Qwen3.5/Qwen3.6** (0.6B, 2B, 4B, 9B) — LLMs
- **Gemma 4** (E2B, E4B) — lightweight LLMs

## wubuwizard Alignment

The wubuwizard project already has the component kernels that Photon 2 fuses:

| Photon 2 Concept | wubuwizard Equivalent |
|-----------------|----------------------|
| Megakernel (fused inference) | SSD-paged MoE (`wubu_ssd_moe.c`) — fuses MoE dispatch + kernel + reduce |
| Operator fusion | `wubu_kernel.c` dispatcher + `wubu_kernel_backends.c` |
| Fast attention | `wubu_fast_attn.c`, `wubu_flash_prefill.c`, `wubu_flashdecode.c` |
| Lazy loading | `wubu_prefix_cache.c`, `wubu_paged_kv.c` |
| Memory pooling | `wubu_arena.c`, `wubu_mem_budget.c` |
| Speculative decoding | `wubu_spec_decode.c`, `wubu_spec_cascade.c` |
| Multi-model serving | `wubu_medusa.c`, `wubu_eagle.c`, `wubu_lmcache.c` |

### Gap Analysis
The wubuwizard has all the individual kernels but lacks:
1. **Compiler-level fusion** — operators are dispatched separately, not fused into
   a single kernel. This is a CUDA/nvcc compilation step, not a C11 kernel.
2. **Zero CPU coordination** — wubuwizard's CPU orchestrator manages kernel
   launches between GPU blocks.

## Recommendations

### For the wubuwizard build:
The SSD-paged MoE system already achieves megakernel-like efficiency for MoE
inference by fusing the expert dispatch + matmul + softmax + reduce into a single
GPU pass (see `src/wubu_ssd_moe.c`). This is the closest wubuwizard analog to
Photon 2's megakernel concept.

For dense (non-MoE) models, the `wubu_kernel.c` dispatcher could be extended
to support kernel fusion via:
- Fusing RMSNorm + Attention + MLP into a single CUDA kernel
- Using `wubu_fast_attn.c` for fused attention computation
- Leveraging `wubu_paged_kv.c` for memory-efficient KV cache management

### No code changes needed:
Photon 2 is an **inference compiler**, not a model architecture. Its concepts
(kernel fusion, single-pass inference) are already partially implemented in
wubuwizard's existing kernel infrastructure. The moondream.ai blog is a reference
report for compiler-level optimizations that could inform future CUDA kernel
development, but there is no C11 kernel to port.

## References
- Photon 2.0: "Inference engine for Physical AI" (Moondream blog, Aug 2025)
- The blog describes the compiler approach, not a new model architecture.
- wubuwizard's `src/wubu_ssd_moe.c` already implements the SSD-paged MoE
  fusion that is the practical equivalent of a megakernel for MoE inference.
