/*
 * wubu_attn_kernels.h -- Attention-side fused kernels (P11/P15/O20).
 */
#ifndef WUBU_ATTN_KERNELS_H
#define WUBU_ATTN_KERNELS_H

/* P11 int2 KV dequant (4 levels/byte, per-block scale/zero). Returns n. */
int wubu_int2_dequant(const unsigned char *packed, int n, float scale, float zero,
                      float *out);

/* P15 fused spec-verify (accept if |draft-ref| <= thr). */
int wubu_spec_verify_fused(float draft_score, float ref_score, float thr);

/* O20 neuro plasticity -> KV quant bits in [bmin,bmax]. */
int wubu_plasticity_bits(float p, int bmin, int bmax);

#endif /* WUBU_ATTN_KERNELS_H */
