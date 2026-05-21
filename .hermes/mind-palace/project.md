# WuBuText AI — Project Overview (May 21, 2026)

## Mission
Build Qwen3.6-35B-A3B inference from scratch in pure C + CUDA with WuBu nested hyperbolic geometry architecture.
**IMPORTANT: Text inference WORKS on CPU.** Vision pipeline WORKS with GPU acceleration. GPU text path is net-negative.

## What Works ✅
- **CPU text inference** — 8.9 tok/s decode, coherent output, 0.9968 cos-sim vs llama.cpp
- **Vision→text pipeline** — 15.7s total (GPU ViT 0.52s, CPU text 6.3s)
- **GPU vision encoder** — 122x faster than CPU (0.52s vs 63.7s)
- **MTP speculative decode** — 8.5 tok/s, 4% acceptance
- **Hybrid GPU/CPU text** — 5.5 tok/s, coherent (net-negative vs CPU)
- **GGUF model loading** — 733+ tensors, 13 quantization types
- **MoE (256 experts, 8 active)** — F32 router + quantized expert matmuls
- **SSM recurrence** — Gated DeltaNet, 30 layers, 128-dim state
- **GQA attention** — 16 Q-heads, 2 KV-heads, IMRoPE
- **CUDA kernels** — batched quant matmul, SSM recurrence, MoE v5, vision ViT
- **All CPU reference tools** — ref_dumper, compare_moe_expert, layer_cos_sim

## What's Not Done 🔲 (P2+)
| Feature | Priority | Notes |
|---------|----------|-------|
| CUDA sm_120 bug docs → skill | P2.0 | Documented in DA v13, formalize |
| Llama.cpp inline hooks | P2.1 | Replace ref_dumper with direct C++ hooks |
| GPU RMSNorm + SiLU kernels | P2.2 | Kernels exist, not wired |
| Chunked prefill | P2.3 | 3-7x speedup at 256K |
| RoPE extrapolation 4x | P2.4 | 64K→256K, single frequency param |
| NSA sparse attention | P2.5 | O(L log L) from DeepSeek-V3.2 |
| Sigmoid gating + load balancing | P2.6 | DeepSeekMoE algorithm |
| FP8 Tensor Cores | P2.7 | sm_120 FP8 dot product |
| Shared experts | P3 | DeepSeekMoE shared + routed |
| N-way hedged spec-decode | P3 | tailslayer/ vault |
| Hamiltonian KV cache (~10x) | P3 | vault/hamilton/ |
| Training pipeline | P4+ | Pure C + CUDA training |
| Multi-modal (audio, video) | P5 | vault/ |

## Hardware
- CPU: Ryzen 7950X (16 cores, DDR5)
- GPU: RTX 5050 (sm_120 Blackwell, 13.1 CUDA, ~2560 cores)
- Storage: NVMe SSD

## Key Achievements
- gguf_raw_size(IQ2_XXS) fix: 72→66 bytes/block — eliminated NaN cascade
- Per-expert dequant: 3.9ms/expert via optimized C grid lookup
- GPU vision: 0.52s for 27 F32 ViT layers (122x vs CPU)
- GPU output projection: cuBLAS SGEMM replaces 2B CPU FMAs
- CUDA sm_120: 3 compiler bugs documented + workarounds
- DA v13: Root cause of GPU MoE divergence identified (fundamental code-path diff)
- All 6 env flags (GPU_SUPPORT, FORCE_CPU_MOE, FORCE_CPU_SSM, etc.) verified working
