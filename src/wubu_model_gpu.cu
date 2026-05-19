/**
 * wubu_model_gpu.cu — GPU-accelerated forward path for wubu_model_t.
 *
 * Provides weight upload, GQA forward, and cleanup — callable from C via
 * extern "C" functions. Compiled with nvcc, linked into gen_text_gpu.
 *
 * Two weight modes:
 *   F32 mode (default): dequant→F32→cuBLAS  (~1GB VRAM for 10 GQA layers)
 *   Quantized mode (future): keep Q5_K on GPU, dequant-on-fly kernel
 */
#include "wubu_model.h"
#include "cuda_kernels.h"
#include "bench.h"
#include "gguf_reader.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

// ================================================================
// Per-layer GPU GQA weights (F32 dequantized for cuBLAS)
// ================================================================
typedef struct {
    float *d_attn_q;       // [D_MODEL, q_dim*2] — fused Q+gate
    float *d_attn_k;       // [D_MODEL, kv_dim]
    float *d_attn_v;       // [D_MODEL, kv_dim]
    float *d_attn_out_w;   // [q_dim, D_MODEL]
    float *d_q_norm_w;     // [HEAD_DIM]
    float *d_k_norm_w;     // [HEAD_DIM]
} gpu_gqa_layer_t;

// ================================================================
// GPU context — stored as opaque pointer in wubu_model_t
// ================================================================
typedef struct {
    // CUDA handles
    cublasHandle_t handle;
    cudaStream_t   stream;
    bool initialized;

    // Weights — per-layer GQA (only 10 layers, not all 40)
    gpu_gqa_layer_t *gqa_weights;  // [n_layers], NULL for SSM layers
    int n_layers;

    // Persistent GPU KV cache (one per GQA layer)
    float **d_k_cache;  // [n_gqa_layers][max_ctx, kv_dim]
    float **d_v_cache;  // [n_gqa_layers][max_ctx, kv_dim]
    int *cache_len;     // current fill length per layer
    int max_ctx;        // max cached positions

    // RoPE sin/cos table
    float *d_sincos;    // [max_ctx, ROTARY_DIM]

    // Scratch buffers (reusable, sized for one chunk)
    float *d_x;         // [chunk_sz, D_MODEL] input
    float *d_scr;       // [chunk_sz, q_dim*2] fused Q+gate
    float *d_ktmp;      // [chunk_sz, kv_dim] K projection
    float *d_vtmp;      // [chunk_sz, kv_dim] V projection
    float *d_gout;      // [chunk_sz, D_MODEL] GQA output
    float *d_score_scr; // chunked attn score scratch
    float *d_qtmp;      // [chunk_sz, q_dim] Q-contiguous buffer for RMSNorm/RoPE/attn
    int chunk_sz;

    // Count of GQA layers (for KV cache indexing)
    int n_gqa_layers;
} gpu_ctx_t;

// ================================================================
// Helpers
// ================================================================
static int count_gqa_layers(const wubu_model_t *model) {
    int n = 0;
    for (int i = 0; i < model->n_layers; i++)
        if (!model->layers[i].is_ssm) n++;
    return n;
}

static int gqa_layer_idx(const wubu_model_t *model, int l) {
    int idx = 0;
    for (int i = 0; i < l; i++)
        if (!model->layers[i].is_ssm) idx++;
    return idx;
}

// ================================================================
// RoPE precomputation (on device, matches infer_text_gpu pattern)
// ================================================================
static void precompute_rotary_host(float *h_sc, int maxT) {
    for (int p = 0; p < maxT; p++) {
        for (int i = 0; i < ROTARY_DIM / 2; i++) {
            // MRoPE sections: [11, 11, 10, 0] — each section restarts freq
            int pair = i;
            int offset_in_section;
            if (pair < 11) {
                offset_in_section = pair;                    // section 0: pairs 0..10
            } else if (pair < 22) {
                offset_in_section = pair - 11;               // section 1: pairs 11..21
            } else {
                offset_in_section = pair - 22;               // section 2: pairs 22..31
            }
            double freq = pow(ROPE_THETA, -2.0 * offset_in_section / ROTARY_DIM);
            double angle = (double)p * freq;
            h_sc[p * ROTARY_DIM + i * 2]     = (float)sin(angle);
            h_sc[p * ROTARY_DIM + i * 2 + 1] = (float)cos(angle);
        }
    }
}

// ================================================================
// Init: upload GQA weights, create KV cache, allocate scratch
// ================================================================
extern "C"
int wubu_model_gpu_init(wubu_model_t *model, int max_ctx, int chunk_sz) {
    if (model->gpu_ctx) {
        fprintf(stderr, "GPU: already initialized\n");
        return 1;
    }

    gpu_ctx_t *gpu = (gpu_ctx_t *)calloc(1, sizeof(gpu_ctx_t));
    if (!gpu) return 0;

    // CUDA context
    cudaError_t ce = cudaSetDevice(0);
    if (ce != cudaSuccess) { free(gpu); return 0; }
    cublasCreate(&gpu->handle);
    cudaStreamCreate(&gpu->stream);
    cublasSetStream(gpu->handle, gpu->stream);
    // Enable TF32
    cublasSetMathMode(gpu->handle, CUBLAS_TF32_TENSOR_OP_MATH);

    gpu->n_layers = model->n_layers;
    gpu->n_gqa_layers = count_gqa_layers(model);
    gpu->max_ctx = max_ctx;
    gpu->chunk_sz = chunk_sz;

    // Allocate GQA weight array (one per layer, NULL for SSM)
    gpu->gqa_weights = (gpu_gqa_layer_t *)calloc(model->n_layers, sizeof(gpu_gqa_layer_t));
    if (!gpu->gqa_weights) { wubu_model_gpu_free(model); return 0; }

    // Find GGUF context with buffered data
    gguf_ctx *ctx = model->gguf_ctx;
    if (!ctx || !ctx->data_blob) {
        fprintf(stderr, "GPU: no GGUF context with buffered data\n");
        wubu_model_gpu_free(model);
        return 0;
    }

    printf("GPU: uploading %d GQA layer weights...\n", gpu->n_gqa_layers);

    const int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;       // 4096
    const int q_dim_x2 = q_dim * 2;                      // 8192
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;      // 512

    for (int l = 0; l < model->n_layers; l++) {
        if (model->layers[l].is_ssm) {
            gpu->gqa_weights[l].d_attn_q = NULL;
            continue;
        }

        gpu_gqa_layer_t *gw = &gpu->gqa_weights[l];
        float *h_buf;
        gguf_tensor_info *t;
        char name[256];

        // Allocate device memory
        gw->d_attn_q     = wubu_cuda_alloc(D_MODEL * q_dim_x2 * sizeof(float));
        gw->d_attn_k     = wubu_cuda_alloc(D_MODEL * kv_dim * sizeof(float));
        gw->d_attn_v     = wubu_cuda_alloc(D_MODEL * kv_dim * sizeof(float));
        gw->d_attn_out_w = wubu_cuda_alloc(q_dim * D_MODEL * sizeof(float));
        gw->d_q_norm_w   = wubu_cuda_alloc(GQA_HEAD_DIM * sizeof(float));
        gw->d_k_norm_w   = wubu_cuda_alloc(GQA_HEAD_DIM * sizeof(float));

        if (!gw->d_attn_q || !gw->d_attn_k || !gw->d_attn_v ||
            !gw->d_attn_out_w || !gw->d_q_norm_w || !gw->d_k_norm_w) {
            fprintf(stderr, "GPU: allocation failed for layer %d\n", l);
            wubu_model_gpu_free(model);
            return 0;
        }

        // attn_q.weight — fused Q+gate
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); wubu_model_gpu_free(model); return 0; }
        h_buf = (float *)malloc(D_MODEL * q_dim_x2 * sizeof(float));
        gguf_read_tensor_f32(ctx, t, h_buf, D_MODEL * q_dim_x2);
        wubu_cuda_to_device(h_buf, gw->d_attn_q, D_MODEL * q_dim_x2 * sizeof(float), gpu->stream);
        free(h_buf);

        // attn_k.weight
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); wubu_model_gpu_free(model); return 0; }
        h_buf = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
        gguf_read_tensor_f32(ctx, t, h_buf, D_MODEL * kv_dim);
        wubu_cuda_to_device(h_buf, gw->d_attn_k, D_MODEL * kv_dim * sizeof(float), gpu->stream);
        free(h_buf);

        // attn_v.weight
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); wubu_model_gpu_free(model); return 0; }
        h_buf = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
        gguf_read_tensor_f32(ctx, t, h_buf, D_MODEL * kv_dim);
        wubu_cuda_to_device(h_buf, gw->d_attn_v, D_MODEL * kv_dim * sizeof(float), gpu->stream);
        free(h_buf);

        // attn_output.weight
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); wubu_model_gpu_free(model); return 0; }
        h_buf = (float *)malloc(q_dim * D_MODEL * sizeof(float));
        gguf_read_tensor_f32(ctx, t, h_buf, q_dim * D_MODEL);
        wubu_cuda_to_device(h_buf, gw->d_attn_out_w, q_dim * D_MODEL * sizeof(float), gpu->stream);
        free(h_buf);

        // attn_q_norm.weight
        snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); wubu_model_gpu_free(model); return 0; }
        h_buf = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
        gguf_read_tensor_f32(ctx, t, h_buf, GQA_HEAD_DIM);
        wubu_cuda_to_device(h_buf, gw->d_q_norm_w, GQA_HEAD_DIM * sizeof(float), gpu->stream);
        free(h_buf);

        // attn_k_norm.weight
        snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); wubu_model_gpu_free(model); return 0; }
        h_buf = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
        gguf_read_tensor_f32(ctx, t, h_buf, GQA_HEAD_DIM);
        wubu_cuda_to_device(h_buf, gw->d_k_norm_w, GQA_HEAD_DIM * sizeof(float), gpu->stream);
        free(h_buf);
    }
    cudaStreamSynchronize(gpu->stream);
    printf("GPU: GQA weights uploaded\n");

    // Allocate persistent GPU KV cache (one per GQA layer)
    gpu->d_k_cache = (float **)calloc(model->n_layers, sizeof(float *));
    gpu->d_v_cache = (float **)calloc(model->n_layers, sizeof(float *));
    gpu->cache_len = (int *)calloc(model->n_layers, sizeof(int));
    if (!gpu->d_k_cache || !gpu->d_v_cache || !gpu->cache_len) {
        wubu_model_gpu_free(model); return 0;
    }
    for (int l = 0; l < model->n_layers; l++) {
        if (model->layers[l].is_ssm) continue;
        size_t cache_bytes = (size_t)max_ctx * kv_dim * sizeof(float);
        gpu->d_k_cache[l] = wubu_cuda_alloc(cache_bytes);
        gpu->d_v_cache[l] = wubu_cuda_alloc(cache_bytes);
        cudaMemset(gpu->d_k_cache[l], 0, cache_bytes);
        cudaMemset(gpu->d_v_cache[l], 0, cache_bytes);
        gpu->cache_len[l] = 0;
    }
    printf("GPU: KV cache allocated (%d layers × %d ctx)\n", gpu->n_gqa_layers, max_ctx);

    // RoPE sin/cos table (host computed, device stored)
    float *h_sc = (float *)malloc((size_t)max_ctx * ROTARY_DIM * sizeof(float));
    precompute_rotary_host(h_sc, max_ctx);
    gpu->d_sincos = wubu_cuda_alloc((size_t)max_ctx * ROTARY_DIM * sizeof(float));
    wubu_cuda_to_device(h_sc, gpu->d_sincos, (size_t)max_ctx * ROTARY_DIM * sizeof(float), gpu->stream);
    free(h_sc);

    // Scratch buffers
    gpu->d_x = wubu_cuda_alloc((size_t)chunk_sz * D_MODEL * sizeof(float));
    gpu->d_scr = wubu_cuda_alloc((size_t)chunk_sz * q_dim_x2 * sizeof(float));
    gpu->d_ktmp = wubu_cuda_alloc((size_t)chunk_sz * kv_dim * sizeof(float));
    gpu->d_vtmp = wubu_cuda_alloc((size_t)chunk_sz * kv_dim * sizeof(float));
    gpu->d_gout = wubu_cuda_alloc((size_t)chunk_sz * D_MODEL * sizeof(float));

    // Score scratch for chunked attention
    size_t score_bytes = wubu_cuda_chunked_attn_query_scratch(chunk_sz, max_ctx);
    gpu->d_score_scr = wubu_cuda_alloc(score_bytes);

    // Q-contiguous buffer: [chunk_sz, q_dim] floats
    gpu->d_qtmp = wubu_cuda_alloc((size_t)chunk_sz * q_dim * sizeof(float));

    cudaStreamSynchronize(gpu->stream);
    gpu->initialized = true;
    model->gpu_ctx = (void *)gpu;

    printf("GPU: init complete (%.1f MB GQA weights)\n",
           (double)gpu->n_gqa_layers * (double)(D_MODEL * (q_dim_x2 + kv_dim*2) + q_dim * D_MODEL) * 4.0 / 1048576.0);
    return 1;
}

// ================================================================
// GPU GQA Forward: process one GQA layer on GPU
// Input:  h_norm [C, D_MODEL] — already RMSNorm'd on CPU
// Output: h_attn [C, D_MODEL] — attention output (will be resid-added on CPU)
// Also appends K,V to persistent GPU KV cache.
// ================================================================
extern "C"
int wubu_model_gpu_gqa_forward(wubu_model_t *model, int layer_idx,
                                const float *h_norm, int C, float *h_attn) {
    gpu_ctx_t *gpu = (gpu_ctx_t *)model->gpu_ctx;
    if (!gpu || !gpu->initialized) return 0;
    if (model->layers[layer_idx].is_ssm) return 0;

    gpu_gqa_layer_t *gw = &gpu->gqa_weights[layer_idx];
    if (!gw->d_attn_q) return 0;

    cublasHandle_t ch = gpu->handle;
    cudaStream_t st = gpu->stream;
    const int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
    const int q_dim_x2 = q_dim * 2;
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;

    // Upload normed input to GPU
    cudaMemcpyAsync(gpu->d_x, h_norm, (size_t)C * D_MODEL * sizeof(float),
                    cudaMemcpyHostToDevice, st);
    cudaStreamSynchronize(st);

    // === Q (fused Q+gate), K, V projections via cuBLAS ===
    // d_scr: [C, q_dim_x2] fused Q+gate
    // d_ktmp: [C, kv_dim] K
    // d_vtmp: [C, kv_dim] V
    wubu_cuda_matmul(ch, gpu->d_x, C, D_MODEL, gw->d_attn_q, q_dim_x2, gpu->d_scr, 1.0f, 0.0f);
    wubu_cuda_matmul(ch, gpu->d_x, C, D_MODEL, gw->d_attn_k, kv_dim, gpu->d_ktmp, 1.0f, 0.0f);
    wubu_cuda_matmul(ch, gpu->d_x, C, D_MODEL, gw->d_attn_v, kv_dim, gpu->d_vtmp, 1.0f, 0.0f);
    cudaStreamSynchronize(st);

    // === Extract Q (strided q_dim_x2) → contiguous d_qtmp (stride q_dim) ===
    wubu_cuda_copy_q_from_fused(gpu->d_qtmp, gpu->d_scr, C, q_dim, st);

    // === Extract gate (strided q_dim_x2) → contiguous in d_scr (overwrite Q part) ===
    // After Q copy, d_scr's first q_dim/token is free. Write gate contiguously there.
    // copy_gate_from_fused: dst[s*qdim+j] = src[s*qdim*2+qdim+j]
    // d_scr[s*8192 + 0..4095] ← d_scr[s*8192 + 4096..8191] — no overlap
    wubu_cuda_copy_gate_from_fused(gpu->d_scr, gpu->d_scr, C, q_dim, st);

    // === RMSNorm Q (contiguous in d_qtmp) ===
    wubu_cuda_rms_norm_heads(C * GQA_Q_HEADS, GQA_HEAD_DIM,
        gpu->d_qtmp, gw->d_q_norm_w, 1e-6f, gpu->d_qtmp, st);

    // === RMSNorm K (contiguous in d_ktmp, stride kv_dim) ===
    wubu_cuda_rms_norm_heads(C * GQA_KV_HEADS, GQA_HEAD_DIM,
        gpu->d_ktmp, gw->d_k_norm_w, 1e-6f, gpu->d_ktmp, st);
    cudaStreamSynchronize(st);

    // === Apply RoPE to Q (d_qtmp) and K (d_ktmp) ===
    int cache_start = gpu->cache_len[layer_idx];
    wubu_cuda_apply_rotary_to_qk(gpu->d_qtmp, gpu->d_ktmp,
        C, C, GQA_Q_HEADS, GQA_KV_HEADS, GQA_HEAD_DIM,
        gpu->d_sincos + (size_t)cache_start * ROTARY_DIM, st);
    cudaStreamSynchronize(st);

    // === Append K, V to persistent cache at position cache_start ===
    cudaMemcpyAsync(gpu->d_k_cache[layer_idx] + (size_t)cache_start * kv_dim,
                    gpu->d_ktmp, (size_t)C * kv_dim * sizeof(float),
                    cudaMemcpyDeviceToDevice, st);
    cudaMemcpyAsync(gpu->d_v_cache[layer_idx] + (size_t)cache_start * kv_dim,
                    gpu->d_vtmp, (size_t)C * kv_dim * sizeof(float),
                    cudaMemcpyDeviceToDevice, st);
    cudaStreamSynchronize(st);

    // Update cache length
    gpu->cache_len[layer_idx] = cache_start + C;

    // === Chunked attention: Q (d_qtmp, contiguous) against all cached K,V ===
    // d_scr now has contiguous gate [C, q_dim] (overwrote Q part from fused buffer)
    // This is safe: chunked_attn reads d_gate_full at stride q_dim per token
    wubu_cuda_chunked_attn(ch, st, C, cache_start + C,
        gpu->d_qtmp,               // Q (RMSNorm'd + RoPE'd, contiguous)
        gpu->d_k_cache[layer_idx], // K_cache (all past positions)
        gpu->d_v_cache[layer_idx], // V_cache (all past positions)
        gpu->d_scr,                // gate (contiguous, stride q_dim)
        gw->d_attn_out_w,          // output projection weight [q_dim, D_MODEL]
        gpu->d_gout,               // output [C, D_MODEL]
        gpu->d_score_scr);         // score scratch

    // Download output back to CPU
    cudaMemcpyAsync(h_attn, gpu->d_gout, (size_t)C * D_MODEL * sizeof(float),
                    cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);

    return 1;
}

// ================================================================
// Free GPU resources
// ================================================================
extern "C"
void wubu_model_gpu_free(wubu_model_t *model) {
    gpu_ctx_t *gpu = (gpu_ctx_t *)model->gpu_ctx;
    if (!gpu) return;

    // Free GQA weights
    if (gpu->gqa_weights) {
        for (int l = 0; l < gpu->n_layers; l++) {
            gpu_gqa_layer_t *gw = &gpu->gqa_weights[l];
            wubu_cuda_free(gw->d_attn_q);
            wubu_cuda_free(gw->d_attn_k);
            wubu_cuda_free(gw->d_attn_v);
            wubu_cuda_free(gw->d_attn_out_w);
            wubu_cuda_free(gw->d_q_norm_w);
            wubu_cuda_free(gw->d_k_norm_w);
        }
        free(gpu->gqa_weights);
    }

    // Free KV cache
    if (gpu->d_k_cache) {
        for (int l = 0; l < gpu->n_layers; l++)
            wubu_cuda_free(gpu->d_k_cache[l]);
        free(gpu->d_k_cache);
    }
    if (gpu->d_v_cache) {
        for (int l = 0; l < gpu->n_layers; l++)
            wubu_cuda_free(gpu->d_v_cache[l]);
        free(gpu->d_v_cache);
    }
    free(gpu->cache_len);

    // Free RoPE table
    wubu_cuda_free(gpu->d_sincos);

    // Free scratch
    wubu_cuda_free(gpu->d_x);
    wubu_cuda_free(gpu->d_scr);
    wubu_cuda_free(gpu->d_ktmp);
    wubu_cuda_free(gpu->d_vtmp);
    wubu_cuda_free(gpu->d_gout);
    wubu_cuda_free(gpu->d_score_scr);
    wubu_cuda_free(gpu->d_qtmp);

    // Destroy CUDA context
    if (gpu->handle) cublasDestroy(gpu->handle);
    if (gpu->stream) cudaStreamDestroy(gpu->stream);

    memset(gpu, 0, sizeof(*gpu));
    free(gpu);
    model->gpu_ctx = NULL;
}
