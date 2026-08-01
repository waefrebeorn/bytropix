/*
 * wubu_misc_gaps.h -- Cross-discipline / OS / neuro gap closers (L05/O12/O13/
 * O15/P12/P13). Opaque-free pure functions (prefault is best-effort).
 */
#ifndef WUBU_MISC_GAPS_H
#define WUBU_MISC_GAPS_H

#include <stddef.h>

/* L05 CacheBlend longest common prefix of two token arrays. */
int wubu_lcp_len(const int *a, const int *b, int n);

/* O12 ProofWright dequant equivalence (quant->dequant within tol). */
int wubu_dequant_equiv(const float *x, int n, float scale, float tol);

/* O13 OS mmap prefault (best-effort warm; -1 if unsupported). */
int wubu_prefault(void *addr, size_t len);

/* O15 neuro rhythmic gate in [0,1] at position p. */
float wubu_rhythmic_gate(int p, float theta, float gamma);

/* P12 KV non-temporal prefetch stream. */
void wubu_kv_prefetch(const float *base, int n, int stride_bytes);

/* P13 fused RoPE rotate + quantize to `bits` (0..2^bits-1) in out[2]. */
void wubu_fused_rope_quant(float x, float y, float angle, int bits, float r,
                           unsigned char *out);

#endif /* WUBU_MISC_GAPS_H */
