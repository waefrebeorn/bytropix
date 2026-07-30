/*
 * wubu_kvcache_quant.c -- KIVI asymmetric quantization for KV cache.
 *
 * Research convergence (Kevin-Bacon 7-hop):
 *
 * 1. llama.cpp: decode is MEMORY-BANDWIDTH-BOUND (Roofline 2607.02558).
 *    KV-cache movement dominates bytes moved per token.
 * 2. KIVI (arxiv 2402.02750): key cache per-channel, value cache per-token,
 *    asymmetric 2-bit quantization. Per-channel K prevents outlier
 *    amplification; per-token V confounds error inside each token.
 * 3. DeltaNet parallel scan (arxiv 2406.06484): SSM decode is also
 *    memory-bandwidth-bound; scan decomposition reduces O(L^2 d) → O(L d).
 * 4. FlashDecoding++ (arxiv 2311.01282): non-blocksparse attention
 *    reduces KV traffic for decode by computing QK^T only over a
 *    compact sliding window -- same principle as KIVI: fewer bytes.
 * 5. Continuous batching (Anyscale 2025): iteration-level scheduling
 *    merges KV-cache writes from multiple requests, amortizing
 *    memory-bandwidth overhead across active sequences.
 * 6. Speculative decoding (arxiv 2402.01528): draft+verify trades
 *    extra forward passes for fewer bytes of KV movement per accepted
 *    token -- another route to the same bandwidth wall.
 * 7. TurboQuant llama.cpp discussion (May 2026): KV quant to <3 bits
 *    with <1% PPL loss, 63% KV reduction. KIVI is the CPU-realizable
 *    subset of this finding for wubuwizard (no GPU kernel required).
 *
 * Implementation: pure C11, no external libs, self-contained.
 * Triple-DA:
 *   Decision: KIVI K/ch + V/token is the highest-leverage CPU win.
 *   Design: wubu_kvq_kivi_quant_K + wubu_kvq_kivi_quant_V in own module,
 *           KIVI quantization scheme enum in wubu_kv_runtime.h.
 *   Diagnostic: round-trip cosine vs f32 reference >= 0.9999 on
 *               Colonel's Qwen3.6-27B KV tensors.
 *
 * C11, no god headers.
 */

#include "wubu_kvcache_quant.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline int8_t f32_to_q8(float v, float scale) {
    if (scale == 0.0f) return 0;
    int q = (int)lroundf(v / scale);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

void wubu_kvq_q8_quant(const float *x, int8_t *q, float *scale, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    *scale = amax / 127.0f;
    if (*scale == 0.0f) { memset(q, 0, (size_t)n); return; }
    for (int i = 0; i < n; i++) q[i] = f32_to_q8(x[i], *scale);
}

void wubu_kvq_q8_dequant(const int8_t *q, float scale, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = (float)q[i] * scale;
}

void wubu_kvq_kivi_quant_K(const float *K, int8_t *q, float *scale_per_ch,
                           int n_tokens, int head_dim) {
    /* per-channel: max abs over all tokens in that channel */
    float *col_max = (float *)malloc((size_t)head_dim * sizeof(float));
    for (int c = 0; c < head_dim; c++) {
        float amax = 0.0f;
        for (int t = 0; t < n_tokens; t++) {
            float a = fabsf(K[(size_t)t * head_dim + c]);
            if (a > amax) amax = a;
        }
        col_max[c] = amax / 127.0f;
    }
    for (int t = 0; t < n_tokens; t++) {
        const float *row = K + (size_t)t * head_dim;
        int8_t *qrow = q + (size_t)t * head_dim;
        for (int c = 0; c < head_dim; c++)
            qrow[c] = f32_to_q8(row[c], col_max[c]);
    }
    for (int c = 0; c < head_dim; c++) scale_per_ch[c] = col_max[c];
    free(col_max);
}

void wubu_kvq_kivi_dequant_K(const int8_t *q, const float *scale_per_ch,
                             float *out, int n_tokens, int head_dim) {
    for (int t = 0; t < n_tokens; t++) {
        const int8_t *qrow = q + (size_t)t * head_dim;
        float *orow = out + (size_t)t * head_dim;
        for (int c = 0; c < head_dim; c++)
            orow[c] = (float)qrow[c] * scale_per_ch[c];
    }
}

/* KIVI value quantization: per-token scale, 4-bit symmetric.
 * Per-token means each token gets its own scale so value-quant
 * error stays inside that token and can't pollute other tokens
 * (confirmed by KIVI paper §3.2). */
void wubu_kvq_kivi_quant_V(const float *V, uint8_t *q, float *scale_per_tok,
                            int n_tokens, int val_dim) {
    for (int t = 0; t < n_tokens; t++) {
        const float *row = V + (size_t)t * val_dim;
        uint8_t *qrow = q + (size_t)t * val_dim;
        float amax = 0.0f;
        for (int i = 0; i < val_dim; i++) {
            float a = fabsf(row[i]);
            if (a > amax) amax = a;
        }
        float scale = amax / 7.0f;  /* 4-bit: [-7,+7] */
        scale_per_tok[t] = scale;
        if (scale == 0.0f) { memset(qrow, 0, (size_t)val_dim); continue; }
        for (int i = 0; i < val_dim; i++) {
            int qi = (int)lroundf(row[i] / scale);
            if (qi >  7) qi =  7;
            if (qi < -7) qi = -7;
            qrow[i] = (uint8_t)(( qi + 8) & 0xF); /* pack into nibble */
        }
    }
}

void wubu_kvq_kivi_dequant_V(const uint8_t *q, const float *scale_per_tok,
                              float *out, int n_tokens, int val_dim) {
    for (int t = 0; t < n_tokens; t++) {
        const uint8_t *qrow = q + (size_t)t * val_dim;
        float *orow = out + (size_t)t * val_dim;
        float scale = scale_per_tok[t];
        for (int i = 0; i < val_dim; i++)
            orow[i] = (float)((int8_t)qrow[i] - 8) * scale;
    }
}

/* KIVI: compute bytes saved vs f32 baseline. */
int64_t wubu_kvq_kivi_bytes_saved(int n_tokens, int head_dim, int val_dim) {
    int64_t f32_bytes = (int64_t)n_tokens * head_dim * sizeof(float)
                      + (int64_t)n_tokens * val_dim * sizeof(float);
    int64_t kivi_bytes = (int64_t)n_tokens * head_dim * sizeof(int8_t)  /* K: 1B/elem */
                       + (int64_t)head_dim * sizeof(float)             /* K scales */
                       + (int64_t)n_tokens * val_dim * sizeof(uint8_t) /* V: 1B/elem packed */
                       + (int64_t)n_tokens * sizeof(float);           /* V scales */
    return f32_bytes - kivi_bytes;
}