/*
 * wubu_gpu_weight_cache.cu — Persistent GPU weight cache for quantized matmul.
 *
 * Maps CPU quantized weight pointers (from GGUF mmap) to GPU device pointers.
 * Weights are uploaded on first access, then reused across decode steps.
 * This eliminates H→D weight transfer per token, enabling GPU-quantized
 * matmul without the full model upload done by wubu_model_gpu_init().
 *
 * A:03 — GPU weight caching for 512K decode acceleration.
 */
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

/* Forward declaration from gpu_quant_matmul.cu */
extern "C" int wubu_cuda_quant_matmul_batched(const float *x, int C,
    const uint8_t *W_q, int quant_type, int n_rows, int n_cols,
    float *y, cudaStream_t stream);

/* Simple open-addressing hash cache for weight pointers. */
#define GPU_WT_CACHE_SIZE 256

struct gpu_wt_entry_t {
    const uint8_t *cpu_ptr;  /* key: GGUF mmap pointer */
    void *gpu_ptr;           /* value: device pointer */
    size_t bytes;            /* size of the weight blob */
    int quant_type;          /* GGML quantization type */
};

static gpu_wt_entry_t g_wt_cache[GPU_WT_CACHE_SIZE];
static int g_wt_cache_count = 0;

extern "C"
int gpu_weight_cache_lookup(const uint8_t *cpu_ptr, int quant_type,
                            void **gpu_ptr_out, size_t *bytes_out) {
    if (!cpu_ptr) return 0;
    /* Simple linear scan (cache is small: ≤256 entries) */
    for (int i = 0; i < g_wt_cache_count; i++) {
        if (g_wt_cache[i].cpu_ptr == cpu_ptr &&
            g_wt_cache[i].quant_type == quant_type) {
            *gpu_ptr_out = g_wt_cache[i].gpu_ptr;
            *bytes_out = g_wt_cache[i].bytes;
            return 1;
        }
    }
    return 0;
}

extern "C"
int gpu_weight_cache_insert(const uint8_t *cpu_ptr, int quant_type,
                            const uint8_t *host_data, size_t bytes) {
    if (!cpu_ptr || !host_data || bytes == 0) return 0;
    if (g_wt_cache_count >= GPU_WT_CACHE_SIZE) {
        /* Cache full — evict LRU (first entry). */
        /* Free the evicted GPU allocation */
        if (g_wt_cache[0].gpu_ptr) cudaFree(g_wt_cache[0].gpu_ptr);
        memmove(&g_wt_cache[0], &g_wt_cache[1],
                (GPU_WT_CACHE_SIZE - 1) * sizeof(gpu_wt_entry_t));
        g_wt_cache_count--;
    }
    void *d_ptr = NULL;
    if (cudaMalloc(&d_ptr, bytes) != cudaSuccess) return 0;
    if (cudaMemcpy(d_ptr, host_data, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_ptr);
        return 0;
    }
    /* Insert at end (most recent) */
    gpu_wt_entry_t *e = &g_wt_cache[g_wt_cache_count++];
    e->cpu_ptr = cpu_ptr;
    e->gpu_ptr = d_ptr;
    e->bytes = bytes;
    e->quant_type = quant_type;
    cudaStreamSynchronize(0);
    return 1;
}

extern "C"
void gpu_weight_cache_clear(void) {
    for (int i = 0; i < g_wt_cache_count; i++) {
        if (g_wt_cache[i].gpu_ptr) cudaFree(g_wt_cache[i].gpu_ptr);
    }
    g_wt_cache_count = 0;
    memset(g_wt_cache, 0, sizeof(g_wt_cache));
}

/* A:03 — GPU-quantized matmul wrapper for proj_matmul in wubu_ssm.c.
 * Checks weight cache; if not cached, uploads quantized weights to GPU,
 * then calls CUDA quantized matmul kernel. Input/output staging is
 * H→D/D→H per call (unavoidable for single-token decode, but the
 * kernel itself runs on GPU which is 5-10x faster for large GEMV). */
extern "C"
int proj_matmul_gpu(const float *x, const uint8_t *W_q, int quant_type,
                    int n_rows, int n_cols, float *out) {
    /* Debug: verify we're being called */
    if (getenv("WUBU_DEBUG")) {
        fprintf(stderr, "[gpu] proj_matmul_gpu: qt=%d rows=%d cols=%d\n",
                quant_type, n_rows, n_cols);
        fflush(stderr);
    }
    void *d_W = NULL;
    size_t w_bytes = 0;
    /* Try cached weight first */
    if (!gpu_weight_cache_lookup(W_q, quant_type, &d_W, &w_bytes)) {
        /* Q4_K: 144 bytes per 256 elements (GGUF spec: 256 elem block) */
        /* Q5_K: 176 bytes per 64 elements  (GGUF spec: 64 elem block)    */
        /* Q6_K: 210 bytes per 64 elements  (GGUF spec: 64 elem block)    */
        if (quant_type == 12) w_bytes = (size_t)n_rows * ((n_cols + 255) / 256) * 144;
        else if (quant_type == 13) w_bytes = (size_t)n_rows * ((n_cols + 63) / 64) * 176;
        else if (quant_type == 14) w_bytes = (size_t)n_rows * ((n_cols + 63) / 64) * 210;
        else w_bytes = (size_t)n_rows * n_cols * 4; /* fallback */
        w_bytes += 4096; /* safety margin for alignment/rounding */

        if (!gpu_weight_cache_insert(W_q, quant_type, W_q, w_bytes)) {
            return 0; /* allocation/transfer failed, fall back to CPU */
        }
        /* Re-lookup to get the GPU pointer */
        if (!gpu_weight_cache_lookup(W_q, quant_type, &d_W, &w_bytes)) {
            return 0;
        }
    }

    /* Now do the actual matmul: x is CPU, d_W is GPU, out is CPU */
    float *d_x = NULL;
    float *d_out = NULL;
    size_t x_bytes = (size_t)n_rows * sizeof(float);
    size_t out_bytes = (size_t)n_cols * sizeof(float);

    int rc = 0;
    if (cudaMalloc(&d_x, x_bytes) != cudaSuccess) goto gpu_fail;
    if (cudaMalloc(&d_out, out_bytes) != cudaSuccess) goto gpu_fail;

    if (cudaMemcpy(d_x, x, x_bytes, cudaMemcpyHostToDevice) != cudaSuccess) goto gpu_fail;

    if (!wubu_cuda_quant_matmul_batched((const float*)d_x, 1, (const uint8_t*)d_W,
            quant_type, n_rows, n_cols, (float*)d_out, 0)) goto gpu_fail;

    if (cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) goto gpu_fail;

    cudaStreamSynchronize(0);
    rc = 1;
gpu_fail:
    if (d_x) cudaFree(d_x);
    if (d_out) cudaFree(d_out);
    return rc;
}
