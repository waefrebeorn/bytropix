#ifndef WUBU_GPU_GEMMA4_H
#define WUBU_GPU_GEMMA4_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
typedef struct g4_gpu_ctx_t g4_gpu_ctx_t;

/* GPU context */
struct g4_gpu_ctx_t {
    cublasHandle_t cublas;
    cudaStream_t stream;

    /* Device buffers */
    float *d_scratch;       /* dequant workspace */
    float *d_q, *d_k, *d_v;
    float *d_attn_out;
    float *d_ffn_gate, *d_ffn_up;
    float *d_hidden;        /* [N, 3840] ping-pong */
    float *d_normed;
    float *d_logits;
    float *d_row_buf;

    /* KV cache on device */
    float *d_k_cache, *d_v_cache;
    int *d_positions;
    int *d_kv_sizes;

    /* Quantized weights on device (uploaded at init) */
    uint8_t *d_token_embd;   /* Q4_0, stays CPU */
    uint8_t *d_layer_weights; /* all layer Q4_K weights concatenated */

    int max_tokens, max_ctx, n_layers;
    uint8_t *d_qweight_row;
    size_t max_row_bytes;
    bool initialized;
};

/* Init / destroy */
g4_gpu_ctx_t* g4_gpu_init(int max_tokens, int max_ctx, int n_layers);
void g4_gpu_destroy(g4_gpu_ctx_t *ctx);

/* GPU forward for the full model (called from g4_model_forward) */
/* Returns 1 on success, 0 on fallback to CPU needed */
int g4_model_forward_gpu(g4_gpu_ctx_t *gpu_ctx,
                          void *model_ptr,
                          const float *embeddings, int B, int T,
                          float *logits);

/* GPU single-token decode: embedding lookup + forward */
int g4_model_decode_gpu(g4_gpu_ctx_t *gpu_ctx, void *model_ptr,
                        int token, float *logits);

/* Individual kernel launches (for the forward function) */
void g4_gpu_rms_norm(g4_gpu_ctx_t *, const float *, const float *, int, int, float *);
void g4_gpu_rope(g4_gpu_ctx_t *, float *, float *, int, int, int, const int *, int, float, const float *, int);
void g4_gpu_sliding_attn(g4_gpu_ctx_t *, const float *, const float *, const float *, int, int, int, int, int, int *, float *, float *, const int *, const int *, float *);
void g4_gpu_full_attn(g4_gpu_ctx_t *, const float *, int, int, int, int, float *, float *, const int *, int, float *);
void g4_gpu_gelu(g4_gpu_ctx_t *, float *, int);
void g4_gpu_mul(g4_gpu_ctx_t *, const float *, const float *, float *, int);
void g4_gpu_add(g4_gpu_ctx_t *, const float *, const float *, float *, int);
void g4_gpu_scale(g4_gpu_ctx_t *, float *, float, int);
void g4_gpu_softcap(g4_gpu_ctx_t *, float *, int, float);
void g4_gpu_sgemm(g4_gpu_ctx_t *, int, int, int, const float *, const float *, float *, bool, bool);
void g4_gpu_h2d(g4_gpu_ctx_t *, const void *, void *, size_t);
void g4_gpu_d2h(g4_gpu_ctx_t *, const void *, void *, size_t);
void g4_gpu_sync(g4_gpu_ctx_t *);
float* g4_gpu_upload_norms(g4_gpu_ctx_t *, const float *, int);

#ifdef __cplusplus
}
#endif

#endif
