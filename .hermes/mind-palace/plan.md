# Plan — Phase 28b DA: F32 Waste Fixed, Now Fix forward_full Bug

## ✅ Completed (this session)
1. F32 dequant upload removed (`#if 0`) — saves ~2.2 GB VRAM
2. ssm_project() fallback uses row_major kernel (fixes N>1 prefill path)
3. CUDA error check on forward_full d_x upload

## 🔴 P0: Fix forward_full C==1 illegal memory access
The SSM GPU decode path crashes with "an illegal memory access" on every SSM layer. This blocks ALL verification work.

**Debug strategy:**
1. Check row_major quant kernel output buffer sizes — verify d_ssm_qkv_out [256, 8192] fits
2. Check fused beta/alpha kernel — verify weight access bounds (W_beta/alpha: [2048, 32])
3. Insert MARK prints before/after each fused kernel in forward_full (C==1 path)
4. Run with 1-token, read which kernel triggers corruption
5. Check: are output buffers (d_ssm_q_all, d_ssm_k_all, d_ssm_v_all, d_ssm_beta_arr, etc.) properly allocated?

**Files to modify:**
- `src/cuda_kernels.cu` — fused kernels and ssm recurrence
- `src/wubu_model_gpu.cu` — forward_full call sites

## 🟡 P1: Fix forward_full C>1 prefill path
cuBLAS error 13 on C>1 prefill. Currently falls through to ssm_project (now correct with row_major). Worth fixing after C==1 works.

**Root cause guess:** The code at line 1072-1074 prints "C>1 path not yet working" and returns 0. This was left as a TODO. The cuBLAS error was from the old F32 path (which we removed).

## 🟡 P2: Build verification pipeline
After forward_full is fixed:
1. Build ref_dumper (needs llama headers — check /home/wubu/bytropix/llama/llama.h)
2. Generate reference layer dumps via ref_dumper
3. Generate our layer dumps via gen_text (CPU path — SLOW but accurate)
4. Run layer_cos_sim to measure per-layer accuracy
5. Enable GPU SSM and compare against CPU path

## 🟢 P3: Profile Phase
After verification:
1. CUDA events: measure SSM GPU time per layer
2. Compare tok/s: GPU SSM vs CPU SSM baseline
3. Profile MoE, GQA, output proj separately
4. Identify true bottlenecks
