/*
 * wubu_kvcache_quant.h -- KV-cache quantization for the attention layers.
 *
 * WHY (Kevin-Bacon meta-analysis convergence):
 *   - HPC Roofline (2607.02558): LLM *decode* is memory-bandwidth-bound; the
 *     KV cache is the dominant per-token memory movement. Reducing KV bytes
 *     directly raises tok/s on CPU.
 *   - DB buffer-pool theory (Apt-Serve / Q-Infer / llm-d): KV is the hot
 *     mutable buffer; quantize + evict like a cache.
 *   - llama.cpp / Intel Xeon study + KIVI paper: Q8_0 KV = 2x smaller,
 *     near-lossless, *faster* when BW-bound; and K and V need DIFFERENT
 *     quantization axes (K per-channel, V per-token).
 *
 * So we implement two schemes, both CPU-realizable, both round-trippable:
 *   - WUBU_KVQ_Q8_0 : block-32 absmax symmetric (llama.cpp q8_0 layout).
 *   - WUBU_KVQ_KIVI : K per-CHANNEL (one scale per head_dim channel),
 *                      V per-TOKEN  (one scale per token) -- the KIVI recipe.
 *
 * The engine stores K and V separately and can pick a different scheme per
 * side (e.g. KIVI-K + q8_0-V), exactly as the research recommends.
 *
 * Memory: q8_0 = 8.0625 bits/elem (block of 32 -> 2 f32 scale/metadata + 32
 * int8 = 272 bits / 32 = 8.5 bits; we store scale as float + q8 = 9 bytes/32
 * for simplicity = 2.25 bytes/elem). KIVI uses 1 float scale per channel/token
 * + int8 data. Both ~4x smaller than fp32, 2x smaller than fp16.
 */
#ifndef WUBU_KVCACHE_QUANT_H
#define WUBU_KVCACHE_QUANT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WUBU_KVQ_F32 = 0,    /* no quantization (baseline) */
    WUBU_KVQ_Q8_0 = 1,   /* block-32 absmax symmetric */
    WUBU_KVQ_KIVI = 2     /* K per-channel, V per-token (KIVI) */
} wubu_kvq_scheme_t;

/* ---- Q8_0 (block of 32) ----
 * Quantize `n` contiguous fp32 values into q8 (int8) + one float scale.
 * scale = max(|x|)/127, q[i] = round(x[i]/scale) clamped to [-128,127]. */
void wubu_kvq_q8_quant(const float *x, int8_t *q, float *scale, int n);

/* Dequantize one q8 block back to fp32. */
void wubu_kvq_q8_dequant(const int8_t *q, float scale, float *out, int n);

/* ---- KIVI ----
 * K: per-CHANNEL quantization. For a K tensor shaped [n_tokens, head_dim],
 * compute one scale per channel (max abs over all tokens in that channel),
 * quantize each channel independently. Stored as q[n_tokens*head_dim] (int8)
 * + scale[head_dim] (float).
 * V: per-TOKEN quantization. One scale per token (max abs over head_dim),
 * quantize each token row independently. Stored as q[n_tokens*head_dim] (int8)
 * + scale[n_tokens] (float). */
void wubu_kvq_kivi_quant_K(const float *K, int8_t *q, float *scale_per_ch,
                            int n_tokens, int head_dim);
void wubu_kvq_kivi_dequant_K(const int8_t *q, const float *scale_per_ch,
                             float *out, int n_tokens, int head_dim);

void wubu_kvq_kivi_quant_V(const float *V, int8_t *q, float *scale_per_tok,
                            int n_tokens, int head_dim);
void wubu_kvq_kivi_dequant_V(const int8_t *q, const float *scale_per_tok,
                             float *out, int n_tokens, int head_dim);

/* Bytes-per-element for a scheme (for capacity planning / DA checks). */
float wubu_kvq_bytes_per_elem(wubu_kvq_scheme_t scheme);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KVCACHE_QUANT_H */
