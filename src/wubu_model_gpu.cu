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
#include "gpu_quant_matmul.h"
#include "gpu_moe_kernel.h"
#include "gpu_ssm_recurrence.h"
#include "wubu_moe.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
// FP16 conversion kernels (defined in cuda_kernels.cu)
__global__ void f32_to_f16_kernel(const float *in, __half *out, int n);
__global__ void f16_to_f32_kernel(const __half *in, float *out, int n);
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

    // Persistent GPU KV cache (one per GQA layer, growable, FP16)
    void **d_k_cache;  // [n_gqa_layers][cache_capacity, kv_dim] __half
    void **d_v_cache;
    int *cache_len;     // current fill length per layer
    int *cache_cap;     // current allocated capacity per layer
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

    // ================================================================
    // SSM quantized weights (kept on GPU in native Q5_K/Q6_K format)
    // ================================================================
    // Per-layer quantized weight data on GPU
    uint8_t *d_attn_qkv_q[40];   // [n_layers] Q5_K attn_qkv
    uint8_t *d_attn_gate_q[40];  // [n_layers] Q5_K attn_gate
    uint8_t *d_ssm_out_q[40];    // [n_layers] Q6_K ssm_out
    int ssm_qkv_type[40];
    int ssm_gate_type[40];
    int ssm_out_type[40];
    int64_t ssm_qkv_col_stride[40];   // bytes per column
    int64_t ssm_gate_col_stride[40];
    int64_t ssm_out_col_stride[40];
    // Small F32 weights
    float *d_ssm_beta[40];      // [D_MODEL, DT_RANK]
    float *d_ssm_alpha[40];
    float *d_ssm_dt_bias[40];   // [DT_RANK]
    float *d_ssm_a[40];         // [DT_RANK]
    float *d_ssm_conv1d[40];    // [CONV_KERNEL, CONV_DIM]
    float *d_ssm_norm[40];      // [SSM_D_STATE]
    // Pre-allocated SSM output buffers (avoid per-call alloc/free)
    float *d_ssm_qkv_out;       // [chunk_sz, CONV_DIM]
    float *d_ssm_z_out;         // [chunk_sz, VALUE_DIM]
    // Pre-allocated MoE buffers (avoid per-call alloc/free for expert weights)
    uint8_t *d_moe_gate;        // [8][gate_bytes_per_expert]
    uint8_t *d_moe_up;          // [8][up_bytes_per_expert]
    uint8_t *d_moe_down;        // [8][down_bytes_per_expert]
    float   *d_moe_out;         // [8][D_MODEL]
    float   *d_moe_weights;     // [8]
    // SSM recurrence persistent state (per layer, updated in-place)
    float **d_ssm_state;        // [n_layers][V_HEADS][D_STATE][D_STATE]
    // SSM recurrence temp buffers
    float *d_ssm_q_all;         // [V_HEADS][D_STATE]
    float *d_ssm_k_all;         // [V_HEADS][D_STATE]
    float *d_ssm_v_all;         // [V_HEADS][D_STATE]
    float *d_ssm_beta_arr;      // [V_HEADS]
    float *d_ssm_gate_arr;      // [V_HEADS]
    float *d_ssm_delta_out;     // [V_HEADS][D_STATE]
    // FP16 scratch for chunked attention (Q + score tile)
    __half *d_hp_scratch;       // [n_q * hd + ATTEN_TILE * C]
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

    // Allocate persistent GPU KV cache (one per GQA layer, growable, FP16)
    // Start small (KV_CACHE_INIT), grow on demand up to max_ctx
    const int kv_cache_init = 4096;
    gpu->d_k_cache = (void**)calloc(model->n_layers, sizeof(void*));
    gpu->d_v_cache = (void**)calloc(model->n_layers, sizeof(void*));
    gpu->cache_len = (int *)calloc(model->n_layers, sizeof(int));
    gpu->cache_cap = (int *)calloc(model->n_layers, sizeof(int));
    if (!gpu->d_k_cache || !gpu->d_v_cache || !gpu->cache_len || !gpu->cache_cap) {
        wubu_model_gpu_free(model); return 0;
    }
    for (int l = 0; l < model->n_layers; l++) {
        if (model->layers[l].is_ssm) continue;
        int initial = (max_ctx < kv_cache_init) ? max_ctx : kv_cache_init;
        gpu->cache_cap[l] = initial;
        size_t cb = (size_t)initial * kv_dim * sizeof(__half);
        gpu->d_k_cache[l] = wubu_cuda_alloc(cb);
        gpu->d_v_cache[l] = wubu_cuda_alloc(cb);
        cudaMemset(gpu->d_k_cache[l], 0, cb);
        cudaMemset(gpu->d_v_cache[l], 0, cb);
        gpu->cache_len[l] = 0;
    }
    printf("GPU: KV cache allocated (%d layers × %d init, max %d ctx)\\n",
           gpu->n_gqa_layers, kv_cache_init, max_ctx);

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

    // FP16 scratch for chunked attention (Q + score tile)
    // q_dim = GQA_Q_HEADS * GQA_HEAD_DIM = 4096, ATTEN_TILE = 4096
    int hp_scratch_elems = 4096 + 4096 * chunk_sz;
    gpu->d_hp_scratch = (__half*)wubu_cuda_alloc((size_t)hp_scratch_elems * sizeof(__half));

    // Q-contiguous buffer: [chunk_sz, q_dim] floats
    gpu->d_qtmp = wubu_cuda_alloc((size_t)chunk_sz * q_dim * sizeof(float));

    // ============================================================
    // Upload SSM quantized weights to GPU (kept in native Q5_K/Q6_K format)
    // ============================================================
    printf("GPU: uploading SSM quantized weights (%d SSM layers)...\n", model->n_layers - gpu->n_gqa_layers);
    {
        int n_ssm_uploaded = 0;
        double total_mb = 0;
        for (int l = 0; l < model->n_layers; l++) {
            if (!model->layers[l].is_ssm) continue;
            wubu_layer_t *layer = &model->layers[l];
            gguf_tensor_info *t;
            char name[256];

            // Initialize to NULL for safe cleanup
            gpu->d_attn_qkv_q[l] = NULL;
            gpu->d_attn_gate_q[l] = NULL;
            gpu->d_ssm_out_q[l] = NULL;
            for (int i = 0; i < 40; i++) gpu->d_ssm_beta[i] = NULL;
            for (int i = 0; i < 40; i++) gpu->d_ssm_alpha[i] = NULL;
            for (int i = 0; i < 40; i++) gpu->d_ssm_dt_bias[i] = NULL;
            for (int i = 0; i < 40; i++) gpu->d_ssm_a[i] = NULL;
            for (int i = 0; i < 40; i++) gpu->d_ssm_conv1d[i] = NULL;
            for (int i = 0; i < 40; i++) gpu->d_ssm_norm[i] = NULL;

            // attn_qkv.weight (Q5_K)
            snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "SSM GPU: missing %s\n", name); continue; }
            int64_t qkv_n_elems = t->dims[0] * (t->n_dims > 1 ? t->dims[1] : 1);
            int64_t qkv_raw = gguf_raw_size(t->ggml_type, qkv_n_elems);
            gpu->ssm_qkv_type[l] = t->ggml_type;
            gpu->ssm_qkv_col_stride[l] = gguf_raw_size(t->ggml_type, t->dims[1] > 0 ? t->dims[1] : t->dims[0]);
            gpu->d_attn_qkv_q[l] = (uint8_t*)wubu_cuda_alloc((size_t)qkv_raw);
            cudaMemcpyAsync(gpu->d_attn_qkv_q[l], (const uint8_t*)ctx->data_blob + t->data_offset,
                           (size_t)qkv_raw, cudaMemcpyHostToDevice, gpu->stream);
            total_mb += qkv_raw / (1024.0 * 1024.0);

            // attn_gate.weight (Q5_K)
            snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "SSM GPU: missing %s\n", name); continue; }
            int64_t gate_n_elems = t->dims[0] * (t->n_dims > 1 ? t->dims[1] : 1);
            int64_t gate_raw = gguf_raw_size(t->ggml_type, gate_n_elems);
            gpu->ssm_gate_type[l] = t->ggml_type;
            gpu->ssm_gate_col_stride[l] = gguf_raw_size(t->ggml_type, t->dims[1] > 0 ? t->dims[1] : t->dims[0]);
            gpu->d_attn_gate_q[l] = (uint8_t*)wubu_cuda_alloc((size_t)gate_raw);
            cudaMemcpyAsync(gpu->d_attn_gate_q[l], (const uint8_t*)ctx->data_blob + t->data_offset,
                           (size_t)gate_raw, cudaMemcpyHostToDevice, gpu->stream);
            total_mb += gate_raw / (1024.0 * 1024.0);

            // ssm_out.weight (Q6_K)
            snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "SSM GPU: missing %s\n", name); continue; }
            int64_t out_n_elems = t->dims[0] * (t->n_dims > 1 ? t->dims[1] : 1);
            int64_t out_raw = gguf_raw_size(t->ggml_type, out_n_elems);
            gpu->ssm_out_type[l] = t->ggml_type;
            gpu->ssm_out_col_stride[l] = gguf_raw_size(t->ggml_type, t->dims[1] > 0 ? t->dims[1] : t->dims[0]);
            gpu->d_ssm_out_q[l] = (uint8_t*)wubu_cuda_alloc((size_t)out_raw);
            cudaMemcpyAsync(gpu->d_ssm_out_q[l], (const uint8_t*)ctx->data_blob + t->data_offset,
                           (size_t)out_raw, cudaMemcpyHostToDevice, gpu->stream);
            total_mb += out_raw / (1024.0 * 1024.0);

            n_ssm_uploaded++;
        }
        cudaStreamSynchronize(gpu->stream);
        printf("GPU: SSM quantized weights uploaded: %d layers, %.0f MB\n", n_ssm_uploaded, total_mb);
    }

    cudaStreamSynchronize(gpu->stream);
    gpu->initialized = true;
    model->gpu_ctx = (void *)gpu;

    // Allocate SSM output buffers (reused per-layer, sized for max chunk)
    gpu->d_ssm_qkv_out = wubu_cuda_alloc((size_t)gpu->chunk_sz * CONV_DIM * sizeof(float));
    gpu->d_ssm_z_out   = wubu_cuda_alloc((size_t)gpu->chunk_sz * VALUE_DIM * sizeof(float));
    printf("GPU: SSM output buffers: %d x %dx%d floats\n", gpu->chunk_sz, CONV_DIM, VALUE_DIM);

    // Initialize GPU MoE lookup tables (IQ2_XXS grid in constant memory)
    wubu_gpu_moe_init();
    printf("GPU: MoE lookup tables initialized\n");

    // Allocate MoE persistent buffers (for 8 active experts)
    const int64_t moe_bytes = 270336;  // IQ2_XXS for D_MODEL*D_FF = 2048*512
    gpu->d_moe_gate    = (uint8_t*)wubu_cuda_alloc((size_t)(8 * moe_bytes));
    gpu->d_moe_up      = (uint8_t*)wubu_cuda_alloc((size_t)(8 * moe_bytes));
    gpu->d_moe_down    = (uint8_t*)wubu_cuda_alloc((size_t)(8 * moe_bytes));
    gpu->d_moe_out     = wubu_cuda_alloc((size_t)(8 * D_MODEL * sizeof(float)));
    gpu->d_moe_weights = wubu_cuda_alloc((size_t)(8 * sizeof(float)));
    printf("GPU: MoE buffers allocated (3x%dKB + %dKB)\\n",
           (int)(8 * moe_bytes / 1024), (int)(8 * D_MODEL * 4 / 1024));

    // Allocate SSM recurrence state: one [V_HEADS][D_STATE][D_STATE] per SSM layer
    gpu->d_ssm_state = (float**)calloc(gpu->n_layers, sizeof(float*));
    for (int l = 0; l < gpu->n_layers; l++) {
        if (model->layers[l].is_ssm) {
            gpu->d_ssm_state[l] = wubu_cuda_alloc(
                (size_t)SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
            cudaMemsetAsync(gpu->d_ssm_state[l], 0,
                (size_t)SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float),
                gpu->stream);
        }
    }
    // SSM recurrence temp buffers
    gpu->d_ssm_q_all    = wubu_cuda_alloc((size_t)SSM_V_HEADS * SSM_D_STATE * sizeof(float));
    gpu->d_ssm_k_all    = wubu_cuda_alloc((size_t)SSM_V_HEADS * SSM_D_STATE * sizeof(float));
    gpu->d_ssm_v_all    = wubu_cuda_alloc((size_t)SSM_V_HEADS * SSM_D_STATE * sizeof(float));
    gpu->d_ssm_beta_arr = wubu_cuda_alloc((size_t)SSM_V_HEADS * sizeof(float));
    gpu->d_ssm_gate_arr = wubu_cuda_alloc((size_t)SSM_V_HEADS * sizeof(float));
    gpu->d_ssm_delta_out= wubu_cuda_alloc((size_t)SSM_V_HEADS * SSM_D_STATE * sizeof(float));
    printf("GPU: SSM recurrence buffers allocated (%d heads × %d state)\n",
           SSM_V_HEADS, SSM_D_STATE);

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

    // === Q (fused Q+gate), K, V projections via cuBLAS ===
    wubu_cuda_matmul(ch, gpu->d_x, C, D_MODEL, gw->d_attn_q, q_dim_x2, gpu->d_scr, 1.0f, 0.0f);
    wubu_cuda_matmul(ch, gpu->d_x, C, D_MODEL, gw->d_attn_k, kv_dim, gpu->d_ktmp, 1.0f, 0.0f);
    wubu_cuda_matmul(ch, gpu->d_x, C, D_MODEL, gw->d_attn_v, kv_dim, gpu->d_vtmp, 1.0f, 0.0f);

    // === Extract Q and gate from fused ===
    wubu_cuda_copy_q_from_fused(gpu->d_qtmp, gpu->d_scr, C, q_dim, st);
    wubu_cuda_copy_gate_from_fused(gpu->d_scr, gpu->d_scr, C, q_dim, st);

    // === RMSNorm Q and K ===
    wubu_cuda_rms_norm_heads(C * GQA_Q_HEADS, GQA_HEAD_DIM,
        gpu->d_qtmp, gw->d_q_norm_w, 1e-6f, gpu->d_qtmp, st);
    wubu_cuda_rms_norm_heads(C * GQA_KV_HEADS, GQA_HEAD_DIM,
        gpu->d_ktmp, gw->d_k_norm_w, 1e-6f, gpu->d_ktmp, st);

    // === Apply RoPE to Q and K ===
    int cache_start = gpu->cache_len[layer_idx];
    wubu_cuda_apply_rotary_to_qk(gpu->d_qtmp, gpu->d_ktmp,
        C, C, GQA_Q_HEADS, GQA_KV_HEADS, GQA_HEAD_DIM,
        gpu->d_sincos + (size_t)cache_start * ROTARY_DIM, st);

    // === Append K, V to persistent cache (F32→FP16) ===
    int total_needed = cache_start + C;
    if (total_needed > gpu->cache_cap[layer_idx]) {
        // Grow cache: double capacity up to max_ctx
        int new_cap = gpu->cache_cap[layer_idx] * 2;
        if (new_cap > gpu->max_ctx) new_cap = gpu->max_ctx;
        if (new_cap < total_needed) new_cap = total_needed;
        size_t new_bytes = (size_t)new_cap * kv_dim * sizeof(__half);
        size_t old_bytes = (size_t)gpu->cache_cap[layer_idx] * kv_dim * sizeof(__half);
        
        void *new_k = wubu_cuda_alloc(new_bytes);
        void *new_v = wubu_cuda_alloc(new_bytes);
        if (!new_k || !new_v) {
            fprintf(stderr, "GPU: KV cache grow failed for layer %d (%d→%d)\\n",
                    layer_idx, gpu->cache_cap[layer_idx], new_cap);
            wubu_cuda_free((float*)new_k);
            wubu_cuda_free((float*)new_v);
            return 0;
        }
        cudaMemcpyAsync(new_k, gpu->d_k_cache[layer_idx], old_bytes,
                        cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync(new_v, gpu->d_v_cache[layer_idx], old_bytes,
                        cudaMemcpyDeviceToDevice, st);
        wubu_cuda_free((float*)gpu->d_k_cache[layer_idx]);
        wubu_cuda_free((float*)gpu->d_v_cache[layer_idx]);
        gpu->d_k_cache[layer_idx] = new_k;
        gpu->d_v_cache[layer_idx] = new_v;
        gpu->cache_cap[layer_idx] = new_cap;
        fprintf(stderr, "GPU: KV cache layer %d grew to %d\\n", layer_idx, new_cap);
    }
    
    // Convert F32 K→FP16, write to cache
    // d_ktmp has F32 K values; convert in-place via d_score_scr as temp
    int n_elems_k = C * kv_dim;
    int block = 256;
    int grid = (n_elems_k + block - 1) / block;
    f32_to_f16_kernel<<<grid, block, 0, st>>>(gpu->d_ktmp,
        (__half*)gpu->d_score_scr, n_elems_k);
    cudaMemcpyAsync((__half*)gpu->d_k_cache[layer_idx] + (size_t)cache_start * kv_dim,
                    gpu->d_score_scr, (size_t)n_elems_k * sizeof(__half),
                    cudaMemcpyDeviceToDevice, st);
    
    // Convert F32 V→FP16, write to cache
    f32_to_f16_kernel<<<grid, block, 0, st>>>(gpu->d_vtmp,
        (__half*)gpu->d_score_scr, n_elems_k);
    cudaMemcpyAsync((__half*)gpu->d_v_cache[layer_idx] + (size_t)cache_start * kv_dim,
                    gpu->d_score_scr, (size_t)n_elems_k * sizeof(__half),
                    cudaMemcpyDeviceToDevice, st);

    // Update cache length
    gpu->cache_len[layer_idx] = cache_start + C;

    // === Chunked attention (FP16 KV cache) ===
    wubu_cuda_chunked_attn_fp16(ch, st, C, cache_start + C,
        gpu->d_qtmp, gpu->d_k_cache[layer_idx], gpu->d_v_cache[layer_idx],
        gpu->d_scr, gw->d_attn_out_w, gpu->d_gout, gpu->d_score_scr,
        gpu->d_hp_scratch);

    // Download output back to CPU
    cudaMemcpyAsync(h_attn, gpu->d_gout, (size_t)C * D_MODEL * sizeof(float),
                    cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);

    return 1;
}

// ================================================================
// GPU SSM hybrid forward: 3 quantized matmuls on GPU, rest on CPU.
// Uploads x, runs Q5_K/Q6_K matmuls, downloads results.
// Uses pre-allocated d_ssm_qkv_out/d_ssm_z_out buffers.
// ================================================================
extern "C"
int wubu_model_gpu_ssm_project(wubu_model_t *model, int layer_idx,
                                const float *h_norm, int C,
                                float *qkv_out, float *z_out,
                                float *ssm_out_out) {
    gpu_ctx_t *gpu = (gpu_ctx_t *)model->gpu_ctx;
    if (!gpu || !gpu->initialized) return 0;
    if (!model->layers[layer_idx].is_ssm) return 0;
    if (model->layers[layer_idx].ssm.attn_qkv_weight_q == NULL) return 0;

    cudaStream_t st = gpu->stream;

    // Upload input to GPU
    cudaMemcpyAsync(gpu->d_x, h_norm, (size_t)C * D_MODEL * sizeof(float),
                    cudaMemcpyHostToDevice, st);

    // === attn_qkv: quantized matmul ===
    // x [D=2048] @ W_qkv [D, C_qkv=8192] → d_qkv [8192]
    wubu_cuda_quant_matmul(gpu->d_x, gpu->d_attn_qkv_q[layer_idx],
        gpu->ssm_qkv_type[layer_idx], D_MODEL, CONV_DIM,
        gpu->d_ssm_qkv_out, NULL, 0, st);

    // === attn_gate: quantized matmul ===
    // x [D=2048] @ W_gate [D, C_gate=4096] → d_z [4096]
    wubu_cuda_quant_matmul(gpu->d_x, gpu->d_attn_gate_q[layer_idx],
        gpu->ssm_gate_type[layer_idx], D_MODEL, VALUE_DIM,
        gpu->d_ssm_z_out, NULL, 0, st);

    // Both matmuls are enqueued on the same stream, so they run sequentially.
    // One sync, then download both results.
    cudaStreamSynchronize(st);

    cudaMemcpyAsync(qkv_out, gpu->d_ssm_qkv_out, (size_t)C * CONV_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, st);
    cudaMemcpyAsync(z_out, gpu->d_ssm_z_out, (size_t)C * VALUE_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, st);

    // ssm_out projection not yet implemented on GPU (handled on CPU for now)
    (void)ssm_out_out;

    cudaStreamSynchronize(st);
    return 1;
}

// GPU MoE expert forward: replaces 8 expert matmuls with GPU quantized kernel
// Input: x_s [D_MODEL], top-8 expert indices + weights
// Output: expert_contribs [8][D_MODEL] filled with weighted results
// Shared expert and router remain on CPU.
extern "C"
void wubu_model_gpu_moe_experts(
    const moe_weights_t *w,
    const float *x_s,
    const int *indices_s,        // [8] expert indices
    const float *weights_s,      // [8] routing weights
    float expert_contribs[8][D_MODEL],
    void *model_ptr)             // wubu_model_t* for CUDA stream access
{
    if (!w->ffn_gate_exps_q) return;
    // Get CUDA stream from model's GPU context
    wubu_model_t *model = (wubu_model_t *)model_ptr;
    gpu_ctx_t *gpu = (gpu_ctx_t *)model->gpu_ctx;
    if (!gpu || !gpu->initialized) return;
    cudaStream_t stream = gpu->stream;

    // Compute per-expert byte sizes
    int64_t gate_bytes = gguf_raw_size(w->ffn_gate_exps_q_type, (int64_t)D_MODEL * D_FF);
    int64_t up_bytes   = gguf_raw_size(w->ffn_up_exps_q_type,   (int64_t)D_MODEL * D_FF);
    int64_t down_bytes = gguf_raw_size(w->ffn_down_exps_q_type, (int64_t)D_FF * D_MODEL);

    // Prepare 8 expert weight data pointers
    const uint8_t *gate_q_ptrs[8], *up_q_ptrs[8], *down_q_ptrs[8];
    float wgts[8];
    int n_active = 0;
    for (int k = 0; k < 8; k++) {
        int e = indices_s[k];
        if (e < 0 || weights_s[k] < 1e-30f) {
            memset(expert_contribs[k], 0, D_MODEL * sizeof(float));
            continue;
        }
        gate_q_ptrs[n_active] = w->ffn_gate_exps_q + (int64_t)e * gate_bytes;
        up_q_ptrs[n_active]   = w->ffn_up_exps_q   + (int64_t)e * up_bytes;
        down_q_ptrs[n_active] = w->ffn_down_exps_q + (int64_t)e * down_bytes;
        wgts[n_active] = weights_s[k];
        n_active++;
    }

    if (n_active == 0) return;

    // Allocate temp output buffer on host (8 experts × D_MODEL)
    float *gpu_out = (float*)calloc(8 * D_MODEL, sizeof(float));
    if (!gpu_out) return;

    // Run GPU MoE — writes each expert's contribution to gpu_out[e*D_MODEL .. (e+1)*D_MODEL-1]
    wubu_gpu_moe_forward_experts(
        x_s,
        (const uint8_t**)gate_q_ptrs, gate_bytes,
        (const uint8_t**)up_q_ptrs, up_bytes,
        (const uint8_t**)down_q_ptrs, down_bytes,
        w->ffn_gate_exps_q_type, w->ffn_up_exps_q_type, w->ffn_down_exps_q_type,
        wgts, gpu_out, stream,
        gpu->d_moe_gate, gpu->d_moe_up, gpu->d_moe_down,
        gpu->d_moe_out, gpu->d_moe_weights);

    // Distribute output across per-expert contribution buffers
    int active_idx = 0;
    for (int k = 0; k < 8; k++) {
        int e = indices_s[k];
        if (e < 0 || weights_s[k] < 1e-30f) continue;
        memcpy(expert_contribs[k], gpu_out + active_idx * D_MODEL,
               D_MODEL * sizeof(float));
        active_idx++;
    }

    free(gpu_out);
}
// Used for the first 2 projections (attn_qkv, attn_gate) and
// separately for the final output projection (ssm_out).
// ================================================================
extern "C"
int wubu_model_gpu_quant_matmul(const float *x, int n_rows, int n_cols,
                                 const uint8_t *d_W_q, int quant_type,
                                 float *y, cudaStream_t stream) {
    float *d_x, *d_y;
    cudaMalloc(&d_x, (size_t)n_rows * sizeof(float));
    cudaMalloc(&d_y, (size_t)n_cols * sizeof(float));
    cudaMemcpyAsync(d_x, x, (size_t)n_rows * sizeof(float), cudaMemcpyHostToDevice, stream);
    int ret = wubu_cuda_quant_matmul(d_x, d_W_q, quant_type, n_rows, n_cols, d_y, NULL, 0, stream);
    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(y, d_y, (size_t)n_cols * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_x);
    cudaFree(d_y);
    return ret;
}

// ================================================================
// Free GPU resources
// ================================================================
extern "C"
void wubu_model_gpu_free(wubu_model_t *model) {
    gpu_ctx_t *gpu = (gpu_ctx_t *)model->gpu_ctx;
    if (!gpu) return;

    // Free SSM quantized weights
    if (gpu->d_attn_qkv_q) {
        for (int i = 0; i < gpu->n_layers; i++) {
            wubu_cuda_free((float*)gpu->d_attn_qkv_q[i]);
            wubu_cuda_free((float*)gpu->d_attn_gate_q[i]);
            wubu_cuda_free((float*)gpu->d_ssm_out_q[i]);
        }
    }

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
            wubu_cuda_free((float*)gpu->d_k_cache[l]);
        free(gpu->d_k_cache);
    }
    if (gpu->d_v_cache) {
        for (int l = 0; l < gpu->n_layers; l++)
            wubu_cuda_free((float*)gpu->d_v_cache[l]);
        free(gpu->d_v_cache);
    }
    free(gpu->cache_len);
    free(gpu->cache_cap);

    // Free RoPE table
    wubu_cuda_free(gpu->d_sincos);

    // Free scratch
    wubu_cuda_free(gpu->d_x);
    wubu_cuda_free((float*)gpu->d_scr);
    wubu_cuda_free((float*)gpu->d_ktmp);
    wubu_cuda_free((float*)gpu->d_vtmp);
    wubu_cuda_free(gpu->d_gout);
    wubu_cuda_free(gpu->d_score_scr);
    wubu_cuda_free((float*)gpu->d_hp_scratch);
    wubu_cuda_free(gpu->d_qtmp);

    // Free SSM output buffers
    wubu_cuda_free(gpu->d_ssm_qkv_out);
    wubu_cuda_free(gpu->d_ssm_z_out);

    // Free MoE buffers
    wubu_cuda_free((float*)gpu->d_moe_gate);
    wubu_cuda_free((float*)gpu->d_moe_up);
    wubu_cuda_free((float*)gpu->d_moe_down);
    wubu_cuda_free(gpu->d_moe_out);
    wubu_cuda_free(gpu->d_moe_weights);

    // Free SSM recurrence state and temp buffers
    if (gpu->d_ssm_state) {
        for (int l = 0; l < gpu->n_layers; l++)
            wubu_cuda_free(gpu->d_ssm_state[l]);
        free(gpu->d_ssm_state);
    }
    wubu_cuda_free(gpu->d_ssm_q_all);
    wubu_cuda_free(gpu->d_ssm_k_all);
    wubu_cuda_free(gpu->d_ssm_v_all);
    wubu_cuda_free(gpu->d_ssm_beta_arr);
    wubu_cuda_free(gpu->d_ssm_gate_arr);
    wubu_cuda_free(gpu->d_ssm_delta_out);

    // Destroy CUDA context
    if (gpu->handle) cublasDestroy(gpu->handle);
    if (gpu->stream) cudaStreamDestroy(gpu->stream);

    memset(gpu, 0, sizeof(*gpu));
    free(gpu);
    model->gpu_ctx = NULL;
}
