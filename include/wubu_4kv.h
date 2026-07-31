#ifndef WUBU_4KV_H
#define WUBU_4KV_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * wubu_4kv.h -- 4-bit KV-cache quantization (SAW-INT4 / TurboQuant approach).
 *
 * Kevin-Bacon 7-hop convergence on KV-cache compression:
 *  - KIVI (2402.02750): K per-channel, V per-token — wubu_kvcache_quant.c
 *  - TurboQuant (ICLR 2026, Google+NYU): <3-bit KV, zero accuracy loss
 *  - SAW-INT4 (2604.19157): block-diagonal Hadamard rotation on K
 *  - Ecco (ISCA 2025): entropy-adaptive per-head adaptive quant
 *  - SGLang (2026): fused rotation-quant kernel, zero e2e overhead
 *
 * All implementations are pure C11, no external libs. The Hadamard rotation
 * is a fixed Sylvester-H matrix (order 8) — no FFT libraries required.
 */

/* Quantize keys with Hadamard BDR rotation → INT4 (nibble-packed in uint8_t).
 * Scale is per-channel (per head_dim). */
void wubu_4kv_quant_K(const float *K, uint8_t *q, float *scale_per_ch,
                       int n_tokens, int head_dim);

/* Dequantize keys: int4 → fp32, then inverse Hadamard (H is self-inverse).
 * inv_h parameter reserved for future non-Hadamard transforms. */
void wubu_4kv_dequant_K(const uint8_t *q, const float *scale_per_ch,
                         const float *inv_h, float *out,
                         int n_tokens, int head_dim);

/* Quantize values → INT4 (nibble-packed, per-token scale). */
void wubu_4kv_quant_V(const float *V, uint8_t *q, float *scale_per_tok,
                       int n_tokens, int val_dim);

/* Dequantize values: int4 → fp32. */
void wubu_4kv_dequant_V(const uint8_t *q, const float *scale_per_tok,
                         float *out, int n_tokens, int val_dim);

/* INT3 value quantization (TurboQuant style, <3 bits).
 * 3-bit symmetric packed 8 values per 3 bytes. */
void wubu_4kv_quant_V3(const float *V, uint8_t *q, float *scale_per_tok,
                         int n_tokens, int val_dim);

void wubu_4kv_dequant_V3(const uint8_t *q, const float *scale_per_tok,
                           float *out, int n_tokens, int val_dim);

/* Ecco-style entropy-adaptive: skip_head[h]=1 keeps head h in FP16 passthrough. */
void wubu_4kv_quant_ecco(const float *V, uint8_t *q, float *scale_per_tok,
                           const uint8_t *skip_head, int n_tokens,
                           int val_dim, int n_heads, int head_dim);

/* KV cache compression ratios. */
int64_t wubu_4kv_bytes_saved(int n_tokens, int head_dim, int val_dim);
int64_t wubu_4kv_bytes_saved_v3(int n_tokens, int head_dim, int val_dim);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_4KV_H */
