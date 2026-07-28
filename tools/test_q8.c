/* Test: wubu_q8 (Area G — Q8_0 lossless weight quant). */
#include "wubu_q8.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define N 1024

int main(void) {
    float *x = (float *)malloc(sizeof(float) * N);
    float *y = (float *)malloc(sizeof(float) * N);
    int8_t *q = (int8_t *)malloc(sizeof(int8_t) * N);
    uint16_t *s = (uint16_t *)malloc(sizeof(uint16_t) * (N / 32));
    srand(99);
    for (int i = 0; i < N; i++) x[i] = ((rand() % 2000) / 1000.0f) - 1.0f;

    wubu_q8_quant(x, q, s, N);
    wubu_q8_dequant(q, s, y, N);
    float c = wubu_q8_cosine(x, y, N);
    printf("Q8_0 cosine sim = %.6f (expect > 0.999)\n", c);
    assert(c > 0.999f);
    free(x); free(y); free(q); free(s);
    printf("ALL Q8 TESTS PASSED\n");
    return 0;
}
