/* Test: wubu_mxfp4 (Round-4 #433 — OCP Microscaling FP4/FP8 round-trip). */
#include "wubu_mxfp4.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

int main(void) {
    int n = 32 * 8;   /* 8 blocks */
    float *x = malloc(sizeof(float)*n);
    for (int i=0;i<n;i++) x[i] = ((i*13)%100)/20.0f - 2.5f;  /* spread of values */
    /* MXFP4 round-trip */
    int bufsz = n/32 * (32/2 + 1);
    uint8_t *buf = malloc(bufsz);
    int wrote = wubu_mxfp4_pack(x, n, buf);
    printf("MXFP4 packed %d floats -> %d bytes (expect %d, ~4.25x shrink)\n", n, wrote, bufsz);
    assert(wrote == bufsz);
    float *y = malloc(sizeof(float)*n);
    wubu_mxfp4_unpack(buf, n, y);
    /* cosine similarity should be high (E2M1 coarseness -> ~0.9+) */
    double dot=0, na=0, nb=0;
    for(int i=0;i<n;i++){ dot+=x[i]*y[i]; na+=x[i]*x[i]; nb+=y[i]*y[i]; }
    double cos = dot / (sqrt(na)*sqrt(nb));
    printf("MXFP4 round-trip cosine = %.4f (expect > 0.90)\n", cos);
    assert(cos > 0.90);

    /* MXFP8 round-trip (finer) */
    int buf8 = n/32 * (32 + 1);
    uint8_t *buf8p = malloc(buf8);
    int w8 = wubu_mxfp8_pack(x, n, buf8p);
    assert(w8 == buf8);
    float *y8 = malloc(sizeof(float)*n);
    wubu_mxfp8_unpack(buf8p, n, y8);
    dot=0; nb=0;
    for(int i=0;i<n;i++){ dot+=x[i]*y8[i]; nb+=y8[i]*y8[i]; }
    cos = dot / (sqrt(na)*sqrt(nb));
    printf("MXFP8 round-trip cosine = %.4f (expect > 0.99)\n", cos);
    assert(cos > 0.99);
    free(buf8p); free(y8);

    /* DA: bad args */
    assert(wubu_mxfp4_pack(x, 33, buf)==-1);   /* not multiple of 32 */
    assert(wubu_mxfp4_unpack(buf, 33, y)==-1);
    free(x); free(y); free(buf);
    printf("ALL MXFP4 TESTS PASSED\n");
    return 0;
}
