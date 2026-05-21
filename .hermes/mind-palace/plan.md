# Plan — Phase 28o: P2 Hardware Utilization & Feature Cream

## 🔴 P0: GPU MoE Hidden State Divergence — COMPLETE
Root cause identified (DA v13): 0.9888 per-layer cos-sim is FUNDAMENTAL — not a single bug. Different code paths produce different IEEE rounding. Hybrid path accepted.

### What Was Done
1. ✅ v5 Q8_K kernel: quantize x to Q8_K, use int8 dot product
2. ✅ CUDA sm_120 bugs: 3 workarounds applied + documented
3. ✅ Per-expert compare tool: compare_moe_expert
4. ✅ DA v13: comprehensive root cause analysis
5. ✅ GPU MoE disabled by default (FORCE_CPU_MOE)
6. ✅ Documentation: GPU hybrid net-negative for text, CPU-only optimal

### What Was Learned
- 0.32% running cos-sim error → flips token selection in 240K vocab
- Hybrid path (GPU SSM/GQA + CPU MoE) produces coherent text at 5.5 tok/s
- Q8_K quantization is correct but doesn't fix the ~1.1% per-layer error
- For 1:1 parity, would need CPU quantized_matmul ported to GPU (3-5 sessions)

## 🟡 P1: MTP Speculative Decode + Vision — COMPLETE
1. ✅ **Build gen_text_mtp** — working at 8.5 tok/s, 4% acceptance (quantized head)
2. ✅ **Vision pipeline** — screenshot→encoder→mmproj→text→logits verified
   - 2 segfault bugs fixed in wubu_vision.c
   - 256×256 → 128 patches × 2048, no NaN, logit range [-10.8, 14.1]
   - test_vision_real builds with GPU_SUPPORT

## Architectural Finding (P0-P1 Verified)
For Qwen3.6-35B IQ2_M on RTX 5050:
- **GPU text inference is net-negative**: H2D/D2H overhead + thermal throttling from GPU init makes hybrid 2-5x slower than CPU-only.
- **GPU vision encoder is the only GPU win**: Pure F32 SGEMM (cuBLAS) for 27 ViT layers + MMProj. 0.52s GPU vs 63.7s CPU.
- **CPU-only is optimal for text**: 8.9 tok/s decode, 17.8 tok/s prefill.

## 🟡 P2: Hardware Utilization & Feature Cream

### P2 Priority
| Priority | Item | Status | Ref | Effort |
|----------|------|--------|-----|--------|
| **P2.0** | **CUDA sm_120 bug skill** — formalize Blackwell workarounds | ✅ In DA v13 | sm_120 Blackwell | Quick |
| **P2.1** | **Llama.cpp inline hooks** — modify llama.cpp source to dump layer-by-layer hidden states + intermediates. Replace ref_dumper (which uses libllama.so API) with direct C++ hooks inside llama_decode() | 🔲 Not started | ~/llama.cpp/ | 1 session |
| **P2.2** | **GPU RMSNorm + SiLU + gated norm kernels** — kernels exist, not wired into pipeline | 🔲 Kernels exist | src/cuda_kernels.cu | Low |
| **P2.3** | **Chunked prefill** — split long prompts into C-sized chunks. 3-7x prefill speedup at 256K. Infrastructure exists (src/wubu_ssm_chunked.c) | 🔲 Infrastructure exists | Qwen2.5-1M §3.3 | Medium |
| **P2.4** | **RoPE extrapolation 4x** — frequency scaling factor <1 to extend 64K→256K. Single parameter change in attention | 🔲 Not started | Qwen2.5-1M §3.1 | Low |
| **P2.5** | **NSA sparse attention** — O(L log L) for GQA layers at 256K. Local window + global positions | 🔲 Not started | DeepSeek-V3.2 §2.1 | High |
| **P2.6** | **Sigmoid gating + load balancing** — normalized sigmoid gating + auxiliary-loss-free dynamic bias adjustment | 🔲 Not started | DeepSeekMoE, DeepSeek-V3 §2.3 | Medium |
| **P2.7** | **FP8 Tensor Cores** — sm_120 FP8 dot product for batched quant matmul. 2x throughput potential | 🔲 Not started | sm_120 ISA | High |

### Architecture Cross-Reference (Vault-Validated)

| P2 Item | Paper/Code | C File | Theory Status |
|---------|-----------|--------|---------------|
| Sigmoid gating | DeepSeekMoE §3, DeepSeek-V3 §2 | moe.c | ✅ Verified in papers |
| Load balancing | DeepSeek-V3 §2.3 | moe.c | ✅ Verified |
| Chunked prefill | Qwen2.5-1M §3.3 | inference.c | ✅ Infrastructure exists |
| RoPE extrapolation | Qwen2.5-1M §3.1 | attention.c | ✅ Simple param change |
| NSA sparse attention | DeepSeek-V3.2 §2.1, vault/attention/ | attention.c | 🟡 Theory exists |
| MTP spec decode | DeepSeek-V3 §2.4 | speculative.c | ✅ Working (4% accept) |

## P3-P6: Vault-Derived Features (unchanged from DA v12)

| Phase | Area | Source | Priority |
|-------|------|--------|----------|
| P3 | N-way hedged speculative decode | vault/tailslayer/ | After P2 |
| P3 | Hamiltonian KV cache (~10× compression) | vault/hamilton/ | After P2 |
| P3 | WuBuSparseAttention, Topological Seq Model | vault/attention/ | After P2 |
| P3 | Rolling hash attention | vault/hash-mind/ | After P2 |
| P4 | Pure C training reference | vault/c-training/ | Training |
| P5 | Text-to-image (VQ-VAE), diffusion, audio | vault/phase3/, diffusion/, audio/ | Post-training |
| P6 | Lean 4 formal verification | vault/lean-proofs/ | Research |
