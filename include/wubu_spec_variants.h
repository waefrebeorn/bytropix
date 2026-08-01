/*
 * wubu_spec_variants.h -- Speculative-decoding variants (M11/M13/M14/L14).
 */
#ifndef WUBU_SPEC_VARIANTS_H
#define WUBU_SPEC_VARIANTS_H

/* M13 spec + KV-quant co-design: writes chosen (K, bits). */
void wubu_spec_kv_codesign(float acceptance, double b_star, int Kmax,
                           int b_lo, int b_hi, int *out_K, int *out_bits);

/* M14 blockwise parallel verify blocks (ceil(K/nb)). */
int wubu_blockwise_verify_blocks(int K, int nb);

/* M11 KV-reuse: 1 if prefix already cached to `pos` (no re-forward). */
int wubu_kv_reuse_ok(int pos, int prefix_len);

/* L14 activation-beam offload: 1 if (recency*importance) < thresh. */
int wubu_offload_decision(float recency, float importance, float thresh);

#endif /* WUBU_SPEC_VARIANTS_H */
