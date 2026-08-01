/*
 * wubu_linear_attn.h -- Linear / recurrent attention hybrids (S01/S03/S04/S05/S07).
 */
#ifndef WUBU_LINEAR_ATTN_H
#define WUBU_LINEAR_ATTN_H

/* S01 Gated DeltaNet state update. */
int wubu_deltanet_update(const float *S, const float *k, const float *v,
                         int d, float beta, float *Sout);
/* S03 Mamba-2 SSM gated decay. */
int wubu_mamba2_update(const float *S, const float *k, const float *v,
                       int d, float A, float b, float *Sout);
/* S04 GLA per-head gate. */
int wubu_gla_update(const float *S, const float *k, const float *v,
                    int d, float g, float *Sout);
/* S05 RetNet/GSA retention. */
int wubu_retnet_update(const float *S, const float *k, const float *v,
                       int d, float gamma, float *Sout);
/* S07 HGRN2/GSA state expansion. */
int wubu_hgrn2_update(const float *S, const float *k, const float *v,
                      int d, float g, float *Sout);

#endif /* WUBU_LINEAR_ATTN_H */
