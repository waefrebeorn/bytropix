/* gpu_gemma4_forward.cu -- Full GPU forward for Gemma 4 12B.
 * Architecture: dual-head-dim ISWA (256/512 HDIM), 48 layers.
 * Strategy: upload Q4_K weights to GPU at init. Per forward:
 *   - Embedding lookup on CPU (Q4_0, small)
 *   - All matmuls: dequant Q4_K on CPU + cuBLAS SGEMM
 *   - Norms, RoPE, attention: GPU kernels
 *   - LM head: GPU dequant + cuBLAS
 * No CPU fallback per layer. */

#include "gpu_gemma4.h"
#include "wubu_gemma4.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* Forward declare g4_rope_apply from wubu_gemma4_model.c (static) -- unused, kept for reference */

#define Q4K_BLOCK_SIZE 144
#define Q4K_N_ELEMS 256
#define QK_K 256

/* ============= Q4_0 dequant kernel (for LM head) ============= */

__global__ void dequant_q4_0_kernel(const uint8_t *W_q, float *W_f32, int64_t n_blocks) {
    int64_t b = blockIdx.x;
    if (b >= n_blocks) return;
    int tid = threadIdx.x;
    if (tid >= 32) return;

    const uint8_t *blk = W_q + b * 18;
    float d = ldexpf((float)((blk[0] & 0xFF) | ((blk[1] & 0x7F) << 8) & 0x3FF) / 1024.0f,
                     ((blk[1] >> 7) & 0x1F) - 15) * (blk[1] & 0x80 ? -1.0f : 1.0f);
    const uint8_t *qs = blk + 2;
    int shift = (tid & 1) ? 0 : 4;
    int val = (qs[tid/2] >> shift) & 0xF;
    W_f32[(size_t)b * 32 + tid] = d * (float)(val - 8);
}

/* ============= GPU RMSNorm ============= */

__global__ void gpu_rms_norm_kernel(const float *x, const float *w, float *y, int n, int N_tok, float eps) {
    int tid = threadIdx.x, tok = blockIdx.x;
    const float *row = x + tok * n;
    float *out = y + tok * n;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) sum += row[i] * row[i];
    __shared__ float s[256];
    s[tid] = sum;
    for (int ss = blockDim.x/2; ss > 0; ss /= 2) { if (tid < ss) s[tid] += s[tid+ss]; __syncthreads(); }
    __syncthreads();
    float rms = rsqrtf(sum / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x) out[i] = row[i] * rms * w[i];
}

__global__ void gpu_rms_norm_q_kernel(const float *x, const float *w, float *y,
                                       int stride, int N_tok, int head_dim, float eps) {
    int tid = threadIdx.x;
    int tok = blockIdx.x;
    int h = blockIdx.y;
    if (tid >= head_dim) return;
    const float *row = x + (size_t)tok * stride + (size_t)h * head_dim;
    float *out = y + (size_t)tok * stride + (size_t)h * head_dim;
    float val = row[tid];
    extern __shared__ float s[];
    s[tid] = val * val;
    __syncthreads();
    for (int ss = blockDim.x/2; ss > 0; ss /= 2) {
        if (tid < ss) s[tid] += s[tid + ss];
        __syncthreads();
    }
    __syncthreads();
    float rms = rsqrtf(s[0] / (float)head_dim + eps);
    out[tid] = val * rms * w[tid];
}

/* ============= GPU RoPE ============= */

__global__ void gpu_rope_kernel(float *q, float *k, int N_tok, int d_head, int q_stride,
                                 const int *pos, int n_rot, float rope_base,
                                 const float *freqs, int is_full) {
    int tok = blockIdx.x, h = blockIdx.y, i = threadIdx.x * 2;
    if (i >= n_rot) return;
    int p = pos[tok];
    float fv = (is_full && freqs && i/2 < d_head/8) ? freqs[i/2] : 1.0f;
    float theta = (float)p / powf(rope_base * fv, (float)i / (float)d_head);
    float ct = __cosf(theta), st = __sinf(theta);
    int base = tok * q_stride + h * d_head;
    float q0 = q[base+i], q1 = q[base+i+1];
    q[base+i]   = q0*ct - q1*st;
    q[base+i+1] = q0*st + q1*ct;
    if (k) {
        float k0 = k[base+i], k1 = k[base+i+1];
        k[base+i]   = k0*ct - k1*st;
        k[base+i+1] = k0*st + k1*ct;
    }
}

/* ============= GPU Sliding Window Attention ============= */

__global__ void gpu_sliding_attn_kernel(const float *q, const float *k_cache, const float *v_cache,
                                         float *out, int N_tok, int n_heads, int n_kv_heads,
                                         int d_head, int window, const int *pos, int kv_size) {
    extern __shared__ float s_scores[];
    int tok = blockIdx.x, hq = blockIdx.y, kv_h = hq % n_kv_heads, tid = threadIdx.x;
    int q_stride = n_heads * d_head, kv_stride = n_kv_heads * d_head;
    int p = pos[tok], kv_end = kv_size - 1;
    if (kv_end > p) kv_end = p;
    int kv_start = (p > window) ? p - window : 0;
    int kv_len = kv_end - kv_start + 1;
    if (kv_len <= 0) {
        for (int d = tid; d < d_head; d += blockDim.x) out[tok*q_stride + hq*d_head + d] = 0.0f;
        return;
    }
    const float *qv = q + tok*q_stride + hq*d_head;
    float *sc = s_scores;
    float scale = rsqrtf((float)d_head), max_s = -1e30f;
    for (int kp = kv_start + tid; kp <= kv_end; kp += blockDim.x) {
        const float *kv = k_cache + kp*kv_stride + kv_h*d_head;
        float s = 0.0f;
        for (int d = 0; d < d_head; d++) s += qv[d] * kv[d];
        s *= scale;
        sc[kp - kv_start] = s;
        if (s > max_s) max_s = s;
    }
    for (int off = blockDim.x/2; off > 0; off /= 2) { if (tid < off) s_scores[tid] = fmaxf(s_scores[tid], s_scores[tid+off]); __syncthreads(); }
    __syncthreads();
    max_s = s_scores[0];
    float sum_e = 0.0f;
    for (int kp = kv_start + tid; kp <= kv_end; kp += blockDim.x) {
        float e = __expf(sc[kp-kv_start] - max_s);
        sc[kp-kv_start] = e;
        sum_e += e;
    }
    s_scores[tid] = sum_e;
    for (int off = blockDim.x/2; off > 0; off /= 2) { if (tid < off) s_scores[tid] += s_scores[tid+off]; __syncthreads(); }
    __syncthreads();
    sum_e = s_scores[0];
    float inv_sum = 1.0f / (sum_e + 1e-10f);
    for (int d = tid; d < d_head; d += blockDim.x) {
        float val = 0.0f;
        for (int kp = kv_start; kp <= kv_end; kp++)
            val += sc[kp-kv_start] * inv_sum * (v_cache + kp*kv_stride + kv_h*d_head)[d];
        out[tok*q_stride + hq*d_head + d] = val;
    }
}

/* ============= GPU Full Attention ============= */

__global__ void gpu_full_attn_kernel(const float *q, const float *k_cache, const float *v_cache,
                                      float *out, int N_tok, int n_heads, int n_kv_heads,
                                      int d_head, const int *pos, int kv_size) {
    extern __shared__ float s_scores[];
    int tok = blockIdx.x, hq = blockIdx.y, kv_h = hq % n_kv_heads, tid = threadIdx.x;
    int q_stride = n_heads * d_head, kv_stride = n_kv_heads * d_head;
    int p = pos[tok], kv_end = kv_size - 1;
    if (kv_end > p) kv_end = p;
    int kv_len = kv_end + 1;
    if (kv_len <= 0) {
        for (int d = tid; d < d_head; d += blockDim.x) out[tok*q_stride+hq*d_head+d] = 0.0f;
        return;
    }
    const float *qv = q + tok*q_stride + hq*d_head;
    float *sc = s_scores;
    float scale = rsqrtf((float)d_head), max_s = -1e30f;
    for (int kp = tid; kp <= kv_end; kp += blockDim.x) {
        const float *kv = k_cache + kp*kv_stride + kv_h*d_head;
        float s = 0.0f;
        for (int d = 0; d < d_head; d++) s += qv[d] * kv[d];
        s *= scale; sc[kp] = s;
        if (s > max_s) max_s = s;
    }
    for (int off = blockDim.x/2; off > 0; off /= 2) { if (tid < off) s_scores[tid] = fmaxf(s_scores[tid], s_scores[tid+off]); __syncthreads(); }
    __syncthreads(); max_s = s_scores[0];
    float sum_e = 0.0f;
    for (int kp = tid; kp <= kv_end; kp += blockDim.x) {
        float e = __expf(sc[kp] - max_s); sc[kp] = e; sum_e += e;
    }
    s_scores[tid] = sum_e;
    for (int off = blockDim.x/2; off > 0; off /= 2) { if (tid < off) s_scores[tid] += s_scores[tid+off]; __syncthreads(); }
    __syncthreads(); sum_e = s_scores[0];
    float inv_sum = 1.0f / (sum_e + 1e-10f);
    for (int d = tid; d < d_head; d += blockDim.x) {
        float val = 0.0f;
        for (int kp = 0; kp <= kv_end; kp++)
            val += sc[kp] * inv_sum * (v_cache + kp*kv_stride + kv_h*d_head)[d];
        out[tok*q_stride + hq*d_head + d] = val;
    }
}

/* ============= GPU GELU ============= */

__global__ void gpu_gelu_kernel(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i], x3 = v*v*v;
    x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * x3)));
}

/* ============= GPU element-wise ops ============= */

__global__ void gpu_mul_kernel(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) c[i] = a[i]*b[i];
}
__global__ void gpu_add_kernel(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) c[i] = a[i]+b[i];
}
__global__ void gpu_scale_kernel(float *x, float s, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i < n) x[i] *= s;
}
__global__ void gpu_softcap_kernel(float *x, int n, float cap) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) { float sc = x[i]/cap; x[i] = tanhf(sc)*cap; }
}

/* ============= cuBLAS SGEMM wrappers ============= */

static void gpu_sgemm_rm_cm(cublasHandle_t cublas, cudaStream_t stream,
                            int M, int N, int K,
                            const float *d_A_row, const float *d_B_col, float *d_C_row) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(cublas, stream);
    /* C_row = A_row (M×K) × B_col (K×N) where A_row row-major, B_col col-major, C_row row-major
     * cuBLAS column-major: C_col = B_col^T × A_col
     *   op(A) = B_col^T (N×K), lda = K (leading dim of B_col in memory)
     *   op(B) = A_col (K×M), ldb = K (leading dim of A_row^T = A_col)
     *   C_col (N×M), ldc = N
     */
    cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                &alpha, d_B_col, K, d_A_row, K, &beta, d_C_row, N);
}

static void gpu_sgemm_cm_cm(cublasHandle_t cublas, cudaStream_t stream,
                            int M, int N, int K,
                            const float *d_A_col, const float *d_B_col, float *d_C_col,
                            bool transA, bool transB) {
    float alpha = 1.0f, beta = 0.0f;
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = transA ? K : M, ldb = transB ? N : K;
    cublasSetStream(cublas, stream);
    cublasSgemm(cublas, opA, opB, M, N, K, &alpha, d_A_col, lda, d_B_col, ldb, &beta, d_C_col, M);
}

/* ============= Working Q4_K CPU dequant (matches test_gpu_dequant3.c) ============= */

static void dequantize_q4_K_block(const uint8_t *block, float *out) {
    uint16_t d_bits, dmin_bits;
    memcpy(&d_bits, block, 2);
    memcpy(&dmin_bits, block + 2, 2);
    int s = (d_bits >> 15) & 1, e = (d_bits >> 10) & 0x1F, m = d_bits & 0x3FF;
    float d = (e == 0) ? ldexpf((float)m / 1024.0f, -14) * (s ? -1.0f : 1.0f)
            : (e == 31) ? (s ? -__builtin_huge_valf() : __builtin_huge_valf())
            : ldexpf(1.0f + (float)m / 1024.0f, e - 15) * (s ? -1.0f : 1.0f);
    s = (dmin_bits >> 15) & 1; e = (dmin_bits >> 10) & 0x1F; m = dmin_bits & 0x3FF;
    float dmin = (e == 0) ? ldexpf((float)m / 1024.0f, -14) * (s ? -1.0f : 1.0f)
               : (e == 31) ? (s ? -__builtin_huge_valf() : __builtin_huge_valf())
               : ldexpf(1.0f + (float)m / 1024.0f, e - 15) * (s ? -1.0f : 1.0f);
    const uint8_t *scales = block + 4;
    const uint8_t *qs = block + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc1, m1, sc2, m2;
        int idx = is;
        if (idx < 4) { sc1 = scales[idx] & 63; m1 = scales[idx + 4] & 63; }
        else { sc1 = (scales[idx+4] & 0xF) | ((scales[idx-4] >> 6) << 4);
               m1  = (scales[idx+4] >>  4) | ((scales[idx  ] >> 6) << 4); }
        idx = is + 1;
        if (idx < 4) { sc2 = scales[idx] & 63; m2 = scales[idx + 4] & 63; }
        else { sc2 = (scales[idx+4] & 0xF) | ((scales[idx-4] >> 6) << 4);
               m2  = (scales[idx+4] >>  4) | ((scales[idx  ] >> 6) << 4); }
        float d1 = d * (float)sc1; float ml1 = dmin * (float)m1;
        float d2 = d * (float)sc2; float ml2 = dmin * (float)m2;
        const uint8_t *bq = qs + j/2;
        for (int l = 0; l < 32; l++) out[j + l]      = d1 * (float)(bq[l] & 0xF) - ml1;
        for (int l = 0; l < 32; l++) out[j + 32 + l] = d2 * (float)(bq[l] >> 4) - ml2;
        is += 2;
    }
}

/* ============= Dequant + SGEMM for Q4_K (CPU dequant + cuBLAS) ============= */

void gpu_matmul_q4k(cudaStream_t stream, cublasHandle_t cublas,
                            float *d_scratch, const uint8_t *d_W,
                            const float *d_x, int M, int N, int K,
                            float *d_y,
                            const uint8_t *h_W) {
    /* Dequantize Q4_K on CPU using gguf_dequantize (proven correct layout) */
    size_t f32_size = (size_t)K * N * sizeof(float);

    float *h_f32 = (float*)malloc(f32_size);
    if (!h_f32) {
        fprintf(stderr, "OOM in gpu_matmul_q4k\n");
        return;
    }

    /* Use exact working gguf_dequantize with correct tensor type */
    gguf_dequantize(h_W, GGML_TYPE_Q4_K, (int64_t)K * N, h_f32);

    /* Debug: print first 16 dequantized weights of FIRST block */
    if (N == 4096 && K == 3840) {
        printf("[DBG GPU] First 16 dequantized weights: ");
        for (int i = 0; i < 16; i++) printf("%.6f ", h_f32[i]);
        printf("\n");
    }

    /* Upload dequantized weights to GPU */
    cudaMemcpyAsync(d_scratch, h_f32, (size_t)K * N * sizeof(float),
                   cudaMemcpyHostToDevice, stream);
    gpu_sgemm_rm_cm(cublas, stream, M, N, K, d_x, d_scratch, d_y);

    free(h_f32);
}

/* ============= Dequant + SGEMM for Q4_0 (GPU dequant + cuBLAS) ============= */

static void gpu_matmul_q4_0(cudaStream_t stream, cublasHandle_t cublas,
                             float *d_scratch, const uint8_t *d_W,
                             const float *d_x, int M, int N, int K,
                             float *d_y) {
    int blocks_per_col = (K + 31) / 32;
    int total_blocks = blocks_per_col * N;
    dequant_q4_0_kernel<<<total_blocks, 32, 0, stream>>>(d_W, d_scratch, total_blocks);
    gpu_sgemm_rm_cm(cublas, stream, M, N, K, d_x, d_scratch, d_y);
}

/* ============= Unified dispatcher ============= */

void gpu_matmul(cudaStream_t stream, cublasHandle_t cublas,
                         float *d_scratch, const uint8_t *d_W,
                         const float *d_x, int M, int N, int K,
                         float *d_y,
                         const uint8_t *h_W, int ggml_type) {
    if (ggml_type == GGML_TYPE_Q4_K) {
        gpu_matmul_q4k(stream, cublas, d_scratch, d_W, d_x, M, N, K, d_y, h_W);
    } else if (ggml_type == GGML_TYPE_Q4_0) {
        gpu_matmul_q4_0(stream, cublas, d_scratch, d_W, d_x, M, N, K, d_y);
    } else {
        /* Fallback: CPU dequant with gguf_dequantize */
        size_t f32_size = (size_t)K * N * sizeof(float);
        float *h_f32 = (float*)malloc(f32_size);
        if (!h_f32) {
            fprintf(stderr, "OOM in gpu_matmul\n");
            return;
        }
        gguf_dequantize(h_W, ggml_type, (int64_t)K * N, h_f32);
        cudaMemcpyAsync(d_scratch, h_f32, (size_t)K * N * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
        gpu_sgemm_rm_cm(cublas, stream, M, N, K, d_x, d_scratch, d_y);
        free(h_f32);
    }
}

/* ============= Weight management ============= */

static int weights_uploaded = 0;
static uint8_t *d_wq[48], *d_wk[48], *d_wv[48], *d_wo[48], *d_wg[48], *d_wu[48], *d_wd[48];
static float *d_norm[48], *d_qnorm[48], *d_knorm[48], *d_pnorm[48], *d_fnorm[48], *d_pfnorm[48];
static float *d_rope_freqs[48];
static uint8_t *d_token_embd = NULL, *d_output_w = NULL;
static float *d_output_norm = NULL;

static void ensure_weights_uploaded(g4_gpu_ctx_t *ctx, g4_model_t *model) {
    if (weights_uploaded) return;
    cudaStream_t s = ctx->stream;

    auto upload_q = [&](const g4_qweight_t *w) -> uint8_t* {
        if (!w->data || w->raw_bytes <= 0) return NULL;
        uint8_t *d; cudaMallocAsync(&d, w->raw_bytes, s);
        cudaMemcpyAsync(d, w->data, w->raw_bytes, cudaMemcpyHostToDevice, s);
        return d;
    };
    auto upload_f = [&](const float *h, int n) -> float* {
        if (!h) return NULL;
        float *d; cudaMallocAsync(&d, n * sizeof(float), s);
        cudaMemcpyAsync(d, h, n * sizeof(float), cudaMemcpyHostToDevice, s);
        return d;
    };

    for (int i = 0; i < model->n_layers; i++) {
        g4_layer_t *l = &model->layers[i];
        d_wq[i] = upload_q(&l->attn_q);
        d_wk[i] = l->share_kv ? NULL : upload_q(&l->attn_k);
        d_wv[i] = (l->share_kv || l->kv_eq) ? NULL : upload_q(&l->attn_v);
        d_wo[i] = upload_q(&l->attn_out);
        d_wg[i] = upload_q(&l->ffn_gate);
        d_wu[i] = upload_q(&l->ffn_up);
        d_wd[i] = upload_q(&l->ffn_down);
        d_norm[i] = upload_f(l->attn_norm_weight, G4_HIDDEN);
        d_qnorm[i] = upload_f(l->attn_q_norm_weight, l->head_dim);
        d_knorm[i] = upload_f(l->attn_k_norm_weight, l->head_dim);
        d_pnorm[i] = upload_f(l->post_attn_norm_weight, G4_HIDDEN);
        d_fnorm[i] = upload_f(l->ffn_norm_weight, G4_HIDDEN);
        d_pfnorm[i] = upload_f(l->post_ffn_norm_weight, G4_HIDDEN);
        if (l->has_rope_freqs) {
            d_rope_freqs[i] = upload_f(l->rope_freqs, l->head_dim / 2);
        } else {
            d_rope_freqs[i] = NULL;
        }
    }

    if (model->token_embd.data) {
        cudaMallocAsync(&d_token_embd, model->token_embd.raw_bytes, s);
        cudaMemcpyAsync(d_token_embd, model->token_embd.data, model->token_embd.raw_bytes, cudaMemcpyHostToDevice, s);
    }
    if (!model->tied_output && model->output.data) {
        cudaMallocAsync(&d_output_w, model->output.raw_bytes, s);
        cudaMemcpyAsync(d_output_w, model->output.data, model->output.raw_bytes, cudaMemcpyHostToDevice, s);
    }
    cudaMallocAsync(&d_output_norm, G4_HIDDEN * sizeof(float), s);
    cudaMemcpyAsync(d_output_norm, model->output_norm_weight, G4_HIDDEN * sizeof(float), cudaMemcpyHostToDevice, s);

    cudaStreamSynchronize(s);
    weights_uploaded = 1;
    printf("[G4_GPU] All weights uploaded to GPU VRAM\n");
}

/* ============= Full GPU forward ============= */

int g4_model_forward_gpu(g4_gpu_ctx_t *ctx, void *model_ptr,
                          const float *embeddings, int B, int T, float *logits) {
    g4_model_t *model = (g4_model_t*)model_ptr;
    const int N = B * T;
    cudaStream_t stream = ctx->stream;
    cublasHandle_t cublas = ctx->cublas;

    ensure_weights_uploaded(ctx, model);

    float *h_x = (float*)malloc((size_t)N * G4_HIDDEN * sizeof(float));
    memcpy(h_x, embeddings, (size_t)N * G4_HIDDEN * sizeof(float));

    cudaMemcpyAsync(ctx->d_hidden, h_x, (size_t)N * G4_HIDDEN * sizeof(float),
                   cudaMemcpyHostToDevice, stream);

    int *h_pos = (int*)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) h_pos[i] = model->current_pos + i;
    cudaMemcpyAsync(ctx->d_positions, h_pos, (size_t)N * sizeof(int),
                   cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    float *d_x = ctx->d_hidden;
    float *d_n = ctx->d_normed;
    float *d_q = ctx->d_q;
    float *d_k = ctx->d_k;
    float *d_v = ctx->d_v;
    float *d_a = ctx->d_attn_out;
    float *d_g = ctx->d_ffn_gate;
    float *d_u = ctx->d_ffn_up;
    float *d_s = ctx->d_scratch;

    for (int il = 0; il < model->n_layers; il++) {
        g4_layer_t *l = &model->layers[il];
        int hd = l->head_dim, qd = l->q_dim, kvd = l->kv_dim, kvh = l->kv_heads;

        gpu_rms_norm_kernel<<<N, 256, 0, stream>>>(d_x, d_norm[il], d_n, G4_HIDDEN, N, G4_RMS_EPS);

        gpu_matmul(stream, cublas, d_s, d_wq[il], d_n, N, qd, G4_HIDDEN, d_q, l->attn_q.data, l->attn_q.ggml_type);

        if (il == 0) {
            float h_dbg[4];
            cudaMemcpyAsync(h_dbg, d_q, 4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            printf("[G4_GPU] DEBUG d_q[0..3] = %.4f %.4f %.4f %.4f\n", h_dbg[0], h_dbg[1], h_dbg[2], h_dbg[3]);
        }

        if (!l->share_kv) {
            gpu_matmul(stream, cublas, d_s, d_wk[il], d_n, N, kvd, G4_HIDDEN, d_k, l->attn_k.data, l->attn_k.ggml_type);
            if (l->kv_eq) {
                cudaMemcpyAsync(d_v, d_k, (size_t)N * kvd * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            } else {
                gpu_matmul(stream, cublas, d_s, d_wv[il], d_n, N, kvd, G4_HIDDEN, d_v, l->attn_v.data, l->attn_v.ggml_type);
            }
        }

        {
            size_t shm_norm = (size_t)hd * sizeof(float);

            dim3 qn_grid(N, G4_HEADS);
            gpu_rms_norm_q_kernel<<<qn_grid, hd, shm_norm, stream>>>(
                d_q, d_qnorm[il], d_q, qd, N, hd, G4_RMS_EPS);

            if (!l->share_kv) {
                dim3 kn_grid(N, kvh);
                gpu_rms_norm_q_kernel<<<kn_grid, hd, shm_norm, stream>>>(
                    d_k, d_knorm[il], d_k, kvd, N, hd, G4_RMS_EPS);
                gpu_rms_norm_kernel<<<N, 256, 0, stream>>>(d_v, NULL, d_v, kvd, N, G4_RMS_EPS);
            }

            {
                dim3 rope_grid(N, qd / hd);
                int rope_threads = hd / 2;
                if (rope_threads > 128) rope_threads = 128;
                if (!l->share_kv) {
                    gpu_rope_kernel<<<rope_grid, rope_threads, 0, stream>>>(
                        d_q, d_k, N, hd, qd, ctx->d_positions, l->n_rot, l->rope_base, d_rope_freqs[il], l->is_full);
                } else {
                    gpu_rope_kernel<<<rope_grid, rope_threads, 0, stream>>>(
                        d_q, NULL, N, hd, qd, ctx->d_positions, l->n_rot, l->rope_base, d_rope_freqs[il], l->is_full);
                }
            }

            if (!l->share_kv) {
                int kv_pos = model->current_pos;
                size_t kv_off = (size_t)il * ctx->max_ctx * G4_MAX_KV_DIM + (size_t)kv_pos * kvd;
                cudaMemcpyAsync(ctx->d_k_cache + kv_off, d_k, (size_t)kvd * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(ctx->d_v_cache + kv_off, d_v, (size_t)kvd * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream);
            }

            static int h_kv_size[48] = {0};
            if (h_kv_size[il] == 0) h_kv_size[il] = N;
            int kv_size = h_kv_size[il];

            if (l->is_full) {
                dim3 grid(N, G4_HEADS);
                size_t shm = (size_t)(kv_size > 0 ? kv_size : 1024) * sizeof(float);
                int threads = hd < 128 ? hd : 128;
                gpu_full_attn_kernel<<<grid, threads, shm, stream>>>(
                    d_q, ctx->d_k_cache, ctx->d_v_cache, d_a,
                    N, G4_HEADS, kvh, hd, ctx->d_positions, kv_size);
            } else {
                dim3 grid(N, G4_HEADS);
                size_t shm = (size_t)G4_SLIDING_WINDOW * sizeof(float);
                int threads = hd < 128 ? hd : 128;
                gpu_sliding_attn_kernel<<<grid, threads, shm, stream>>>(
                    d_q, ctx->d_k_cache, ctx->d_v_cache, d_a,
                    N, G4_HEADS, kvh, hd, G4_SLIDING_WINDOW, ctx->d_positions, kv_size);
            }

            h_kv_size[il] = kv_size + N;
        }

        gpu_matmul(stream, cublas, d_s, d_wo[il], d_a, N, G4_HIDDEN, qd, d_n, l->attn_out.data, l->attn_out.ggml_type);

        gpu_rms_norm_kernel<<<N, 256, 0, stream>>>(d_n, d_pnorm[il], d_n, G4_HIDDEN, N, G4_RMS_EPS);
        gpu_add_kernel<<<(N*G4_HIDDEN+255)/256, 256, 0, stream>>>(d_x, d_n, d_x, N*G4_HIDDEN);

        gpu_rms_norm_kernel<<<N, 256, 0, stream>>>(d_x, d_fnorm[il], d_n, G4_HIDDEN, N, G4_RMS_EPS);
        gpu_matmul(stream, cublas, d_s, d_wg[il], d_n, N, G4_FFN, G4_HIDDEN, d_g, l->ffn_gate.data, l->ffn_gate.ggml_type);
        gpu_matmul(stream, cublas, d_s, d_wu[il], d_n, N, G4_FFN, G4_HIDDEN, d_u, l->ffn_up.data, l->ffn_up.ggml_type);
        gpu_gelu_kernel<<<(N*G4_FFN+255)/256, 256, 0, stream>>>(d_g, N*G4_FFN);
        gpu_mul_kernel<<<(N*G4_FFN+255)/256, 256, 0, stream>>>(d_u, d_g, d_u, N*G4_FFN);
        gpu_matmul(stream, cublas, d_s, d_wd[il], d_u, N, G4_HIDDEN, G4_FFN, d_n, l->ffn_down.data, l->ffn_down.ggml_type);

        gpu_rms_norm_kernel<<<N, 256, 0, stream>>>(d_n, d_pfnorm[il], d_n, G4_HIDDEN, N, G4_RMS_EPS);
        gpu_add_kernel<<<(N*G4_HIDDEN+255)/256, 256, 0, stream>>>(d_x, d_n, d_x, N*G4_HIDDEN);

        if (l->has_out_scale) {
            gpu_scale_kernel<<<(N*G4_HIDDEN+255)/256, 256, 0, stream>>>(d_x, l->layer_out_scale, N*G4_HIDDEN);
        }
    }

    gpu_rms_norm_kernel<<<N, 256, 0, stream>>>(d_x, d_output_norm, d_n, G4_HIDDEN, N, G4_RMS_EPS);

    uint8_t *d_out_w = d_output_w ? d_output_w : d_token_embd;
    gpu_matmul_q4_0(stream, cublas, d_s, d_out_w, d_n, N, G4_VOCAB, G4_HIDDEN, ctx->d_logits);

    gpu_softcap_kernel<<<(N*G4_VOCAB+255)/256, 256, 0, stream>>>(ctx->d_logits, N*G4_VOCAB, G4_SOFTCAP);

    cudaMemcpyAsync(logits, ctx->d_logits, (size_t)N * G4_VOCAB * sizeof(float),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    model->current_pos += N;
    free(h_x);
    free(h_pos);

    return 1;
}

/* ============= GPU single-token decode ============= */

int g4_model_decode_gpu(g4_gpu_ctx_t *ctx, void *model_ptr, int token, float *logits) {
    g4_model_t *model = (g4_model_t*)model_ptr;
    float embd[G4_HIDDEN];
    int row_bytes = (int)(model->token_embd.raw_bytes / G4_VOCAB);
    if (token < 0 || token >= G4_VOCAB) token = 0;
    gguf_dequantize(model->token_embd.data + (size_t)token * row_bytes,
                   model->token_embd.ggml_type, G4_HIDDEN, embd);
    float scale = sqrtf((float)G4_HIDDEN);
    for (int i = 0; i < G4_HIDDEN; i++) embd[i] *= scale;
    return g4_model_forward_gpu(ctx, model_ptr, embd, 1, 1, logits);
}