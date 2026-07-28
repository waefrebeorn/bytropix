/* Test: wubu_ssm_scan (Area F — chunkwise SSM scan). */
#include "wubu_ssm_scan.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

int main(void) {
    int T = 64, D = 4, C = 16;
    float *A = (float *)malloc(sizeof(float) * D);
    float *Bx = (float *)malloc(sizeof(float) * T * D);
    float *state = (float *)calloc(T * D, sizeof(float));
    srand(7);
    for (int d = 0; d < D; d++) A[d] = 0.9f - 0.1f * d;  /* stable A in (0,1) */
    for (int i = 0; i < T * D; i++) Bx[i] = ((rand() % 100) / 50.0f) - 1.0f;

    float err = wubu_ssm_scan_chunked(A, Bx, state, T, D, C);
    printf("chunkwise SSM scan max err vs serial = %.2e (expect ~0)\n", err);
    assert(err < 1e-5f);

    free(A); free(Bx); free(state);
    printf("ALL SSM-SCAN TESTS PASSED\n");
    return 0;
}
