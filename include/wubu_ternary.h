/*
 * wubu_ternary.h -- BitNet/ternary-weight kernels (T01/T02/T03).
 */
#ifndef WUBU_TERNARY_H
#define WUBU_TERNARY_H

/* T03 absmax scale. */
float wubu_ternary_scale(const float *w, int n);
/* T01 pack ternary (2-bit/val) into bytes. */
int wubu_ternary_pack(const float *w, int n, float scale, float thr, unsigned char *out);
/* dequant packed -> F32. */
int wubu_ternary_unpack(const unsigned char *packed, int n, float scale, float *out);
/* T02 mpGEMV: y = scale * sum ternary_w * act. */
int wubu_mpgemv(const unsigned char *tw, int rows, int cols, float scale,
                const float *act, float *y);

#endif /* WUBU_TERNARY_H */
