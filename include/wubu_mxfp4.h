#ifndef WUBU_MXFP4_H
#define WUBU_MXFP4_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_MX_BLOCK 32

/* OCP Microscaling: pack/unpack MXFP4 (E2M1 + E8M0) and MXFP8 (E4M3 + E8M0),
 * 32-element blocks. n must be a multiple of 32. */
int wubu_mxfp4_pack(const float *x, int n, uint8_t *out);
int wubu_mxfp4_unpack(const uint8_t *in, int n, float *out);
int wubu_mxfp8_pack(const float *x, int n, uint8_t *out);
int wubu_mxfp8_unpack(const uint8_t *in, int n, float *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MXFP4_H */
