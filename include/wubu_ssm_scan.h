#ifndef WUBU_SSM_SCAN_H
#define WUBU_SSM_SCAN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Chunkwise SSM selective scan. state[T*D] in/out. Returns max abs error
 * vs the serial reference (0 when correct). */
float wubu_ssm_scan_chunked(const float *A, const float *Bx, float *state,
                            int T, int D, int C);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SSM_SCAN_H */
