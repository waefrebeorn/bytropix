# Overnight Map — May 19, 2026 PM (Phase 16 Part 2: GPU SSM Recurrence Done)

## Active GPU Components
| Phase | Component | GPU? | Status |
|-------|-----------|------|--------|
| 15 | GQA forward (QKV, RoPE, attention) | 😴 prefill only (N>1) | shipped |
| 16a | SSM matmuls (qkv, gate) | 😴 prefill only (N>16) | shipped |
| 16b | SSM recurrence (selective scan) | 🆕 Verify cos-sim=1.0 | shipped |
| 17 | MoE routed expert compute | 🆕 GPU IQ2_XXS kernel | shipped |
| 13 | Output projection (Q4_K) | ✅ Full GPU SGEMM | shipped |
| — | SSM conv, norm, gated norm | ❌ CPU | next target |
| — | MoE router + shared expert | ❌ CPU | fast enough |
| — | GQA scores, attention softmax | ❌ CPU | fast enough |

## Decode Speed (first run, no thermal throttle)
- CPU (gen_text): 7.3 tok/s — all CPU, no GPU
- GPU (gen_text_gpu with GPU enabled for decode): 6.4 tok/s — GPU only helps prefill
- **For single-token decode, GPU offload overhead > benefit.** SSM recurrence GPU kernel works (cos-sim=1.0) but transfer/sync overhead dominate.

## Key GPU Recurrence Details
- Kernel: `ssm_recurrence_kernel` in `src/gpu_ssm_recurrence.cu`
- 32 V-heads × 128 threads, state [128][128] in global memory (64KB/head)
- Shared memory: 2.5KB/block (q,k,v,hk,diff vectors)
- Cos-sim 1.0 vs CPU reference, max err 1e-6
- Each thread manages one row; uses shared mem for dot product diffusion
- Wired via `w->gpu_ssm_state` check in `wubu_ssm_forward` with `goto gpu_rec_done`
- GPU init guards: `#ifdef GPU_SUPPORT` in wubu_ssm.c, wubu_moe.c

## KV Cache Memory Issue
- 256k context KV cache = 10 GB (10 layers × 2 × 512kB × 256k)
- RTX 5050: 8151 MB total → overcommits, causes swap degradation
- Workaround: `MAX_CTX=4096` env var for testing
- Need: growable KV cache or 256k sparse attention

## GPU GQA Fix
- Fused `attn_qkv.weight` vs separate Q/K/V issue resolved (was stale binary)
- Model: GQA on layers 3,7,11,...,39 (every 4th, offset 3)
- Removed 4 unnecessary `cudaStreamSynchronize` calls
- N>1 threshold: GPU GQA only for prefill

## Next Direction
1. **Port SSM conv + norm + gated norm to GPU** — largest potential speedup for prefill
2. **Growable GPU KV cache** — allocate as needed, not max_ctx upfront
3. **256k context sparse GQA attention** — needed for long context
4. **Evaluate unified forward kernel** — fuse SSM steps for single-kernel decode
