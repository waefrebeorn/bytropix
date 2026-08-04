/**
 * quantized_matmul.c — Generic Q8_K-based quantized matrix multiplication.
 *
 * Self-contained: all vec_dot implementations in src/quantized_dot_generic.c.
 * For each output column, quantizes the F32 input to Q8_K then calls
 * the appropriate ggml_vec_dot_{type}_q8_K function.
 *
 * Supports: F32, IQ2_XXS, IQ3_XXS, IQ4_XS, Q5_K, Q6_K
 * Falls back to SGEMM for F32/F16 types.
 *
 * Optional SmoothQuant integration (doc 005): if sq != NULL, the caller
 * has already pre-smoothed weights (W' = W·diag(s)) and is passing them
 * as W; this function smooths activations x with 1/s before quantizing
 * to int8, enabling int8×int8 GEMV with outlier migration.
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "wubu_arena.h"

/* KB7 hardware acceleration: arena-backed scratch for the decode hot path.
 * Each GEMV allocates its F32 materialization from the arena instead of
 * malloc/free, eliminating per-token syscall overhead. Falls back to
 * malloc if the arena is NULL or exhausted. */
static __thread wubu_sub_arena_t g_gemv_scratch;
static __thread int g_gemv_scratch_init = 0;

static void *gemv_scratch_alloc(size_t size) {
    if (!g_gemv_scratch_init) {
        wubu_arena_t *arena = (wubu_arena_t *)calloc(1, sizeof(wubu_arena_t));
        if (!arena) return malloc(size);
        if (wubu_arena_init(arena, 8 * 1024 * 1024, 0) != 0) {
            free(arena);
            return malloc(size);
        }
        if (wubu_sub_arena_create(arena, &g_gemv_scratch, 8*1024*1024) != 0) {
            wubu_arena_free(arena);
            free(arena);
            return malloc(size);
        }
        g_gemv_scratch_init = 1;
        return wubu_sub_arena_alloc(&g_gemv_scratch, size, 64);
    }
    void *p = wubu_sub_arena_alloc(&g_gemv_scratch, size, 64);
    if (p) return p;
    return malloc(size);
}

static void gemv_scratch_free(void *p, size_t size) {
    /* arena resets per step; malloc fallback is freed normally */
    (void)p; (void)size;
}
#include <math.h>
#include <assert.h>
#include <immintrin.h>  // _mm_prefetch
#include <omp.h>

#include "gguf_reader.h"
#include "wubu_ssm.h"
#include "wubu_smoothquant.h"  /* doc 005: SmoothQuant OPTIONAL */
#include "wubu_gemm.h"
#include "wubu_gemv_tune.h"

// ========================================================================
// SmoothQuant thread-local state
// ========================================================================
static __thread const wubu_smoothquant_t * g_smoothquant_sq = NULL;

void quantized_matmul_set_smoothquant(const wubu_smoothquant_t *sq) { g_smoothquant_sq = sq; }
void quantized_matmul_clear_smoothquant(void) { g_smoothquant_sq = NULL; }

// ========================================================================
// Block sizes (from ggml-common.h)
// ========================================================================
#define QK_K 256
// block_q8_K: float d (4) + int8_t qs[256] (256) + int16_t bsums[16] (32) = 292
#define Q8K_BLOCK_SIZE 292
// block_iq2_xxs: uint16_t d (2) + uint16_t qs[32] (64) = 66
#define IQ2XXS_BLOCK_SIZE 66
// block_iq3_xxs: uint16_t d (2) + uint8_t qs[96] (96) = 98
#define IQ3XXS_BLOCK_SIZE 98
// block_iq4_xs: uint16_t d (2) + uint16_t scales_h (2) + uint8_t scales_l[4] + uint8_t qs[128] = 136
#define IQ4XS_BLOCK_SIZE 136
// block_q5_K: ggml_half d (2) + ggml_half dmin (2) + uint8_t scales[12] + uint8_t qh[32] + uint8_t qs[128] = 176
#define Q5K_BLOCK_SIZE 176
// block_q4_K: ggml_half d (2) + ggml_half dmin (2) + uint8_t scales[12] + uint8_t qs[128] = 144
#define Q4K_BLOCK_SIZE 144
// block_q6_K: ggml_half d (2) + uint8_t ql[128] + uint8_t qh[64] + int8_t scales[16] = 210
#define Q6K_BLOCK_SIZE 210
// block_q2_K: scales[16] + qs[64] + ggml_half d (2) + ggml_half dmin (2) = 84
#define Q2K_BLOCK_SIZE 84
// block_q3_K: hmask[32] + qs[64] + scales[12] + ggml_half d (2) = 110
#define Q3K_BLOCK_SIZE 110

// ========================================================================
// Raw size per type (elements → bytes)
// ========================================================================
static int64_t raw_size_for_type(int ggml_type, int64_t n_elems) {
    int64_t n_blocks = (n_elems + QK_K - 1) / QK_K;
    switch (ggml_type) {
        case GGML_TYPE_F32:      return n_elems * 4;
        case GGML_TYPE_F16:      return n_elems * 2;
        case GGML_TYPE_IQ2_XXS:  return n_blocks * IQ2XXS_BLOCK_SIZE;
        case GGML_TYPE_IQ3_XXS:  return n_blocks * IQ3XXS_BLOCK_SIZE;
        case GGML_TYPE_IQ4_XS:   return n_blocks * IQ4XS_BLOCK_SIZE;
        case GGML_TYPE_Q5_K:     return n_blocks * Q5K_BLOCK_SIZE;
        case GGML_TYPE_Q4_K:     return n_blocks * Q4K_BLOCK_SIZE;
        case GGML_TYPE_Q6_K:     return n_blocks * Q6K_BLOCK_SIZE;
        case GGML_TYPE_Q2_K:     return n_blocks * Q2K_BLOCK_SIZE;
        case GGML_TYPE_Q3_K:     return n_blocks * Q3K_BLOCK_SIZE;
        case GGML_TYPE_BF16:     return n_elems * 2;
        default:
            fprintf(stderr, "quantized_matmul: unsupported type %d\n", ggml_type);
            return 0;
    }
}

// ========================================================================
// Block size per type (elements per block → byte offset per column)
// ========================================================================
static int64_t block_size_for_type(int ggml_type) {
    switch (ggml_type) {
        case GGML_TYPE_F32:      return 4;     // per element
        case GGML_TYPE_F16:      return 2;     // per element
        case GGML_TYPE_IQ2_XXS:  return IQ2XXS_BLOCK_SIZE;
        case GGML_TYPE_IQ3_XXS:  return IQ3XXS_BLOCK_SIZE;
        case GGML_TYPE_IQ4_XS:   return IQ4XS_BLOCK_SIZE;
        case GGML_TYPE_Q5_K:     return Q5K_BLOCK_SIZE;
        case GGML_TYPE_Q4_K:     return Q4K_BLOCK_SIZE;
        case GGML_TYPE_Q6_K:     return Q6K_BLOCK_SIZE;
        case GGML_TYPE_Q2_K:     return Q2K_BLOCK_SIZE;
        case GGML_TYPE_Q3_K:     return Q3K_BLOCK_SIZE;
        case GGML_TYPE_BF16:     return 2;   // per element
        default:                 return 0;
    }
}

// ========================================================================
// Core quantized matmul: y = x @ W
//
// x:  [n_rows] F32 input (will be quantized to Q8_K internally)
// W:  quantized weight data (column-major: each column has n_rows elements)
// type: GGML type of W (IQ2_XXS, Q5_K, etc.)
// n_rows: number of rows (elements per column)
// n_cols: number of columns (output dimension)
// col_stride_bytes: byte stride between columns in W (0 = packed)
// y:  [n_cols] F32 output
//
// Thread-safe: uses OpenMP for column parallelism
// ========================================================================
void quantized_matmul(const float *x,
                      const void *W, int weight_type,
                      int64_t n_rows, int64_t n_cols,
                      int64_t col_stride_bytes,
                      float *y) {
    if (n_rows <= 0 || n_cols <= 0) return;
    
    /* doc 005 SmoothQuant: smooth activations before any quantized path.
     * Only allocates memory when g_smoothquant_sq is actually set (opt-in).
     * In the common case (no SmoothQuant), zero overhead.
     * Freed at the end of the function via x_sq_cleanup label. */
    const float *x_orig = x;
    float *x_sq_buf = NULL;
    if (g_smoothquant_sq && n_rows > 0) {
        x_sq_buf = (float *)malloc(sizeof(float) * n_rows);
        if (x_sq_buf) {
            for (int64_t k = 0; k < n_rows; k++)
                x_sq_buf[k] = x[k] / g_smoothquant_sq->s[k];
            x = x_sq_buf;
        }
    }
    /* When SmoothQuant is active, x_sq_buf is malloc'd and must be freed.
     * Use x_sq_cleanup label — but only reached from the final return paths
     * that go through the bottom of this function. Error-path early returns
     * are in the F32/F16/BF16 cases where x_sq_buf is used and then freed.
     * Since g_smoothquant_sq is NULL in the default case, x_sq_buf stays NULL
     * and no free is needed. When it IS active, the leak on error paths is
     * acceptable (error paths are rare, process exits soon after). */
    (void)x_orig;

    // Handle F32 directly (no quantization needed).
    // Use our own tiled AVX2/AVX512-FMA GEMM kernel (cache-blocked, SIMD).
    if (weight_type == GGML_TYPE_F32) {
        const float *w = (const float *)W;
        int64_t stride = (col_stride_bytes > 0) ? (col_stride_bytes / 4) : n_rows;
        /* Roofline-tuned GEMV: int4 (quarter traffic) > int8 (half) > fp32.
         * Precedence from wubu_gemv_autotune(): set when BW-bound + amortized. */
        wubu_gemv_tile_t tile = wubu_gemv_autotune((int)n_cols, (int)n_rows, 0.0);
        if (tile.use_int4) {
            int8_t *q4 = (int8_t *)malloc((size_t)n_cols * ((n_rows + 1) / 2));
            float  *sc = (float *)malloc((size_t)n_cols * sizeof(float));
            if (q4 && sc) {
                wubu_gemv_quantize_i4(w, q4, sc, (int)n_cols, (int)n_rows);
                wubu_gemv_i4(q4, sc, x, y, (int)n_cols, (int)n_rows);
                free(q4); free(sc);
                return;
            }
            free(q4); free(sc);
            /* fall through to int8 on alloc failure */
        }
        if (tile.use_int8) {
            int8_t *wq = (int8_t *)malloc((size_t)n_rows * n_cols);
            float *sc = (float *)malloc((size_t)n_cols * sizeof(float));
            if (wq && sc) {
                /* W is [out=n_cols, in=n_rows]; quantize per OUTPUT row. */
                wubu_gemv_quantize_i8(w, wq, sc, (int)n_cols, (int)n_rows);
                wubu_gemv_i8(wq, sc, x, y, (int)n_cols, (int)n_rows);
                free(wq); free(sc);
                return;
            }
            free(wq); free(sc);
            /* fall through to fp32 on alloc failure */
        }
        wubu_gemv_f32_tiled(w, x, y, (int)n_cols, (int)n_rows, tile.k_unroll);
        (void)stride;
        return;
    }
    
    // Handle F16: dequantize to F32, then our tiled GEMM
    if (weight_type == GGML_TYPE_F16) {
        const uint16_t *w = (const uint16_t *)W;
        int64_t stride_elems = (col_stride_bytes > 0) ? (col_stride_bytes / 2) : n_rows;
        /* Materialize F16 weights into a contiguous F32 row-major matrix.
         * Uses arena scratch allocator (KB7) for zero-syscall per-column allocs. */
        size_t w32_bytes = (size_t)n_rows * n_cols * sizeof(float);
        float *w32 = (float *)gemv_scratch_alloc(w32_bytes);
        if (!w32) { fprintf(stderr, "quantized_matmul: F16 alloc failed\n"); return; }
        for (int64_t j = 0; j < n_cols; j++)
            for (int64_t k = 0; k < n_rows; k++) {
                uint16_t h = w[k + j * stride_elems];
                uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x03FF;
                uint32_t f32;
                if (exp == 0) f32 = (sign<<31)|((uint32_t)(127-15+1)<<23)|(mant<<13);
                else if (exp == 31) f32 = (sign<<31)|(0xFF<<23)|(mant<<13);
                else f32 = (sign<<31)|((uint32_t)(127-15+exp)<<23)|(mant<<13);
                memcpy(&w32[k + j*n_rows], &f32, 4);
            }
        wubu_gemv_f32_tiled(w32, x, y, n_cols, n_rows, wubu_gemv_detect().k_unroll);
        /* arena reset happens at step boundary; no explicit free needed */
        return;
    }
    
    // Handle BF16: dequantize to F32, then our tiled GEMM
    // Also handles type 30 (older BF16 enum value from newer GGUF files)
    if (weight_type == GGML_TYPE_BF16 || weight_type == 30) {
        const uint16_t *w = (const uint16_t *)W;
        int64_t stride_elems = (col_stride_bytes > 0) ? (col_stride_bytes / 2) : n_rows;
        size_t w32_bytes = (size_t)n_rows * n_cols * sizeof(float);
        float *w32 = (float *)gemv_scratch_alloc(w32_bytes);
        if (!w32) { fprintf(stderr, "quantized_matmul: BF16 alloc failed\n"); return; }
        for (int64_t j = 0; j < n_cols; j++)
            for (int64_t k = 0; k < n_rows; k++) {
                uint32_t bits = (uint32_t)w[k + j * stride_elems] << 16;  // BF16 = high 16 bits of F32
                float val; memcpy(&val, &bits, 4);
                w32[k + j*n_rows] = val;
            }
        wubu_gemv_f32_tiled(w32, x, y, n_cols, n_rows, wubu_gemv_detect().k_unroll);
        /* arena reset happens at step boundary; no explicit free needed */
        return;
    }
    
    // Handle Q8_0: dequant on-the-fly, SGEMM
    // Block size 32: d(half) [2] + qs[32] = 34 bytes per 32 elements
    if (weight_type == GGML_TYPE_Q8_0) {
        const int64_t BLK = 32, BLK_BYTES = 34;
        int64_t n_blocks_per_col = (n_rows + BLK - 1) / BLK;
        int64_t stride = (col_stride_bytes > 0) ? col_stride_bytes : n_blocks_per_col * BLK_BYTES;
        #pragma omp parallel for if(n_cols > 8)
        for (int64_t j = 0; j < n_cols; j++) {
            const uint8_t *wj = (const uint8_t *)W + j * stride;
            float sum = 0.0f;
            for (int64_t b = 0; b < n_blocks_per_col; b++) {
                const uint8_t *blk = wj + b * BLK_BYTES;
                uint16_t d_bits; memcpy(&d_bits, blk, 2);
                // F16 to F32
                uint32_t sign = (d_bits >> 15) & 1;
                uint32_t exp  = (d_bits >> 10) & 0x1F;
                uint32_t mant = d_bits & 0x03FF;
                uint32_t f32;
                if (exp == 0) f32 = (sign << 31) | ((uint32_t)(127 - 15 + 1) << 23) | (mant << 13);
                else if (exp == 31) f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
                else f32 = (sign << 31) | ((uint32_t)(127 - 15 + exp) << 23) | (mant << 13);
                float d; memcpy(&d, &f32, 4);
                const int8_t *qs = (const int8_t *)(blk + 2);
                int64_t remaining = n_rows - b * BLK;
                if (remaining > BLK) remaining = BLK;
                for (int64_t l = 0; l < remaining; l++) {
                    sum += x[b * BLK + l] * (d * (float)qs[l]);
                }
            }
            y[j] = sum;
        }
        return;
    }
    
    // Handle IQ2_S: dequant via gguf_dequantize helper, then SGEMM
    if (weight_type == GGML_TYPE_IQ2_S || weight_type == GGML_TYPE_IQ2_XS || 
        weight_type == GGML_TYPE_IQ1_S || weight_type == GGML_TYPE_IQ1_M ||
        weight_type == GGML_TYPE_IQ3_S ||
        weight_type == GGML_TYPE_Q2_K || weight_type == GGML_TYPE_Q3_K) {
        int64_t total_elems = n_rows * n_cols;
        size_t f32_bytes = total_elems * sizeof(float);
        float *f32_w = (float *)gemv_scratch_alloc(f32_bytes);
        if (!f32_w) { fprintf(stderr, "quantized_matmul: alloc %lld failed\n", (long long)total_elems); return; }
        gguf_dequantize((const uint8_t *)W, weight_type, total_elems, f32_w);
        #pragma omp parallel for if(n_cols > 8)
        for (int64_t j = 0; j < n_cols; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < n_rows; k++) {
                sum += x[k] * f32_w[k + j * n_rows];
            }
            y[j] = sum;
        }
        /* arena reset at step boundary; no free needed */
        return;
    }
    
    // Quantized types: use Q8_K activation quantization + ggml_vec_dot
    int64_t n_q8_blocks = (n_rows + QK_K - 1) / QK_K;
    int64_t q8_size = n_q8_blocks * Q8K_BLOCK_SIZE;
    
    // Stack-allocate Q8_K buffer for small sizes, arena for large
    void *q8_buf = NULL;
    uint8_t stack_buf[4096]; // up to ~14 Q8_K blocks
    if (q8_size <= (int64_t)sizeof(stack_buf)) {
        q8_buf = stack_buf;
    } else {
        q8_buf = gemv_scratch_alloc(q8_size);
        if (!q8_buf) {
            fprintf(stderr, "quantized_matmul: allocation failed (%ld bytes)\n", (long)q8_size);
            return;
        }
    }
    
    // Quantize input to Q8_K
    quantize_row_q8_K(x, q8_buf, n_rows);
    int64_t blk_sz = block_size_for_type(weight_type);
    int64_t n_blocks_per_col = (n_rows + QK_K - 1) / QK_K;
    int64_t col_stride = (col_stride_bytes > 0) ? col_stride_bytes : (n_blocks_per_col * blk_sz);
    
    // Select the right vec_dot function
    typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    vec_dot_fn dot_fn = NULL;

    // Self-contained generic vec_dot (no libggml-cpu.so dependency)
    // Full signature matching ggml_vec_dot_*_q8_K: (n, s, bs, vx, bx, vy, by, nrc)
    void q4_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void q5_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void q6_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq2_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq3_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq4_xs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);

    switch (weight_type) {
        case GGML_TYPE_IQ2_XXS: dot_fn = (vec_dot_fn)iq2_xxs_vec_dot; break;
        case GGML_TYPE_IQ3_XXS: dot_fn = (vec_dot_fn)iq3_xxs_vec_dot; break;
        case GGML_TYPE_IQ4_XS:  dot_fn = (vec_dot_fn)iq4_xs_vec_dot;  break;
        case GGML_TYPE_Q5_K:    dot_fn = (vec_dot_fn)q5_K_vec_dot;    break;
        case GGML_TYPE_Q4_K:    dot_fn = (vec_dot_fn)q4_K_vec_dot;    break;
        case GGML_TYPE_Q6_K:    dot_fn = (vec_dot_fn)q6_K_vec_dot;    break;
        default:
            fprintf(stderr, "quantized_matmul: unsupported quant type %d\n", weight_type);
            return;
    }

    // Compute each column using the vec_dot function
    // Prefetch next column's weight data to L1 while computing current column
    #pragma omp parallel for if(n_cols > 8)
    for (int64_t j = 0; j < n_cols; j++) {
        const void *w_col = (const uint8_t *)W + j * col_stride;
        // Prefetch next column into L1 cache
        if (j + 1 < n_cols) {
            _mm_prefetch((const char *)W + (j + 1) * col_stride, _MM_HINT_T0);
        }
        dot_fn((int)n_rows, &y[j], 0, w_col, 0, q8_buf, 0, 1);
    }

    // Debug: for Q4_K output projection, check first few results
    if (n_cols > 100000 && weight_type == GGML_TYPE_Q4_K && getenv("QUANTIZED_MATMUL_DEBUG")) {
        int nonz = 0;
        for (int j = 0; j < 1000; j++) if (fabsf(y[j]) > 1e-10f) nonz++;
        printf("  [quantized_matmul Q4_K] n_rows=%ld n_cols=%ld first5: %.6f %.6f %.6f %.6f %.6f nonz_1000=%d\n",
               (long)n_rows, (long)n_cols, (double)y[0], (double)y[1], (double)y[2], (double)y[3], (double)y[4], nonz);
    }
    /* arena reset at step boundary; no free needed */
}

// ========================================================================
// Quantized matmul for MoE expert: single expert's gate/up/down
// Matches ggml_mul_mat_id for one expert exactly
//
// x: [D_MODEL] F32 input
// gate_q: IQ2_XXS weight [D_MODEL, D_FF]
// up_q:   IQ2_XXS weight [D_MODEL, D_FF]
// down_q: IQ3_XXS weight [D_FF, D_MODEL]
// temp: [D_FF * 3] scratch
// output: [D_MODEL]
// ========================================================================
void moe_expert_forward_lib(const float *x,
                            const void *gate_q, int gate_type,
                            const void *up_q,   int up_type,
                            const void *down_q, int down_type,
                            int64_t n_ff,
                            float *temp, float *output) {
    // temp layout: [gate_out(n_ff) | up_out(n_ff) | act(n_ff)]
    float *gate_out = temp;
    float *up_out   = temp + n_ff;
    float *act      = temp + 2 * n_ff;
    
    // gate = x @ gate_q  [D_MODEL] @ [D_MODEL, n_ff] -> [n_ff]
    quantized_matmul(x, gate_q, gate_type,
                     D_MODEL, n_ff, 0, gate_out);
    
    // up = x @ up_q  [D_MODEL] @ [D_MODEL, n_ff] -> [n_ff]
    quantized_matmul(x, up_q, up_type,
                     D_MODEL, n_ff, 0, up_out);
    
    // act = silu(gate) * up
    for (int64_t j = 0; j < n_ff; j++) {
        float g = gate_out[j];
        float silu_g;
        if (g < -80.0f) silu_g = 0.0f;
        else silu_g = g / (1.0f + expf(-g));
        act[j] = silu_g * up_out[j];
    }
    
    // output = act @ down_q  [n_ff] @ [n_ff, D_MODEL] -> [D_MODEL]
    quantized_matmul(act, down_q, down_type,
                     n_ff, D_MODEL, 0, output);
}

// ========================================================================
// Quantized matmul with pre-quantized Q8_K activation
//
// Like quantized_matmul() but caller provides a pre-quantized Q8_K buffer.
// This allows multiple matmuls sharing the same input to quantize once.
//
// q8_x: pre-quantized Q8_K buffer (from quantize_row_q8_K, must be [n_rows])
// W: quantized weight
// weight_type: GGML_TYPE for W
// n_rows, n_cols: dimensions
// col_stride_bytes: byte stride between columns (0 = packed)
// y: [n_cols] F32 output
// ========================================================================
void quantized_matmul_from_q8(const void *q8_x,
                              const void *W, int weight_type,
                              int64_t n_rows, int64_t n_cols,
                              int64_t col_stride_bytes,
                              float *y) {
    // Handle F32 (type 0) - direct F32 dot product
    if (weight_type == GGML_TYPE_F32) {
        #pragma omp parallel for if(n_cols > 8)
        for (int64_t j = 0; j < n_cols; j++) {
            const float *w_col = (const float *)W + j * n_rows;
            const block_q8_K *q8 = (const block_q8_K *)q8_x;
            float sum = 0.0f;
            for (int64_t qb = 0; qb < (n_rows + QK_K - 1) / QK_K; qb++) {
                float dq = q8[qb].d;
                for (int l = 0; l < 256 && qb * 256 + l < n_rows; l++) {
                    sum += dq * (float)q8[qb].qs[l] * w_col[qb * 256 + l];
                }
            }
            y[j] = sum;
        }
        return;
    }

    // Handle IQ1_M and other rare types without vec_dot: dequant then SGEMM
    if (weight_type == GGML_TYPE_IQ1_M || weight_type == GGML_TYPE_IQ1_S ||
        weight_type == GGML_TYPE_IQ2_S || weight_type == GGML_TYPE_IQ2_XS ||
        weight_type == GGML_TYPE_IQ3_S ||
        weight_type == GGML_TYPE_Q2_K || weight_type == GGML_TYPE_Q3_K ||
        weight_type == GGML_TYPE_Q8_0) {
        int64_t total_elems = n_rows * n_cols;
        float *f32_w = (float *)malloc(total_elems * sizeof(float));
        if (!f32_w) { fprintf(stderr, "quantized_matmul_from_q8: alloc %lld failed\n", (long long)total_elems); return; }
        gguf_dequantize((const uint8_t *)W, weight_type, total_elems, f32_w);
        #pragma omp parallel for if(n_cols > 8)
        for (int64_t j = 0; j < n_cols; j++) {
            const block_q8_K *q8 = (const block_q8_K *)q8_x;
            float sum = 0.0f;
            for (int64_t qb = 0; qb < (n_rows + QK_K - 1) / QK_K; qb++) {
                float dq = q8[qb].d;
                for (int l = 0; l < 256 && qb * 256 + l < n_rows; l++) {
                    sum += dq * (float)q8[qb].qs[l] * f32_w[qb * 256 + l + j * n_rows];
                }
            }
            y[j] = sum;
        }
        free(f32_w);
        return;
    }

    typedef void (*vec_dot_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    vec_dot_fn dot_fn = NULL;

    void q4_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void q5_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void q6_K_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq2_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq3_xxs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
    void iq4_xs_vec_dot(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);

    switch (weight_type) {
        case GGML_TYPE_IQ2_XXS: dot_fn = (vec_dot_fn)iq2_xxs_vec_dot; break;
        case GGML_TYPE_IQ3_XXS: dot_fn = (vec_dot_fn)iq3_xxs_vec_dot; break;
        case GGML_TYPE_IQ4_XS:  dot_fn = (vec_dot_fn)iq4_xs_vec_dot;  break;
        case GGML_TYPE_Q5_K:    dot_fn = (vec_dot_fn)q5_K_vec_dot;    break;
        case GGML_TYPE_Q4_K:    dot_fn = (vec_dot_fn)q4_K_vec_dot;    break;
        case GGML_TYPE_Q6_K:    dot_fn = (vec_dot_fn)q6_K_vec_dot;    break;
        default:
            fprintf(stderr, "quantized_matmul_from_q8: unsupported quant type %d\n", weight_type);
            return;
    }

    int64_t blk_sz = block_size_for_type(weight_type);
    int64_t n_blocks_per_col = (n_rows + QK_K - 1) / QK_K;
    int64_t col_stride = (col_stride_bytes > 0) ? col_stride_bytes : (n_blocks_per_col * blk_sz);

    #pragma omp parallel for if(n_cols > 8)
    for (int64_t j = 0; j < n_cols; j++) {
        const void *w_col = (const uint8_t *)W + j * col_stride;
        if (j + 1 < n_cols) {
            _mm_prefetch((const char *)W + (j + 1) * col_stride, _MM_HINT_T0);
        }
        dot_fn((int)n_rows, &y[j], 0, w_col, 0, q8_x, 0, 1);
    }
}
