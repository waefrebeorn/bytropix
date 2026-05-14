/**
 * wubu-cuda.cuh — CUDA kernel declarations for WuBu ops
 * Matches llama.cpp's .cuh pattern (one per op file).
 */

#ifndef WUBU_CUDA_H
#define WUBU_CUDA_H

#include "ggml-cuda/common.cuh"
#include "ggml.h"

/* ─── Poincaré Exponential Map ─── */
void ggml_cuda_op_wubu_exp_map(ggml_backend_cuda_context& ctx, ggml_tensor* dst);

/* ─── Poincaré Logarithmic Map ─── */
void ggml_cuda_op_wubu_log_map(ggml_backend_cuda_context& ctx, ggml_tensor* dst);

/* ─── Top-2 MoE Routing ─── */
void ggml_cuda_op_wubu_moe_top2(ggml_backend_cuda_context& ctx, ggml_tensor* dst);

/* ─── Rolling Hash (SimpleHash encoder) ─── */
void ggml_cuda_op_wubu_rolling_hash(ggml_backend_cuda_context& ctx, ggml_tensor* dst);

/* ─── Möbius Addition ─── */
void ggml_cuda_op_wubu_mobius_add(ggml_backend_cuda_context& ctx, ggml_tensor* dst);

/* ─── Quaternion Hamilton Product ─── */
void ggml_cuda_op_wubu_hamilton_product(ggml_backend_cuda_context& ctx, ggml_tensor* dst);

#endif /* WUBU_CUDA_H */
