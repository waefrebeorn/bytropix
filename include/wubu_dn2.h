/*
 * wubu_dn2.h -- Gated DeltaNet-2 (S02) + ternary STE proxy (T04).
 */
#ifndef WUBU_DN2_H
#define WUBU_DN2_H

/* S02 Gated DeltaNet-2 decoupled erase/write update. */
int wubu_dn2_update(const float *S, const float *k, const float *v,
                    int d, float beta, float erase_g, float write_g, float *Sout);
/* T04 ternary STE forward + grad-passes flag. */
int wubu_ternary_ste(float x, float thr, float *out_tern, int *grad_passes);

#endif /* WUBU_DN2_H */
