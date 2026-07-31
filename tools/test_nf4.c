/* test_nf4.c — NormalFloat 4-bit quantization correctness test
 * Compares NF4 roundtrip vs Q8 and F32 for normally-distributed data.
 */
#include "wubu_nf4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float vec_norm(float *v, int n) {
    float s=0; for(int i=0;i<n;i++) s+=v[i]*v[i]; return sqrtf(s);
}

int main(void) {
    printf("=== NormalFloat 4-bit Quantization Test ===\n\n");
    int d = 128 * 32;  /* 4096 elements = 128 NF4 blocks */
    int n_blocks = d / 32;

    /* Generate normally-distributed weights (Box-Muller) */
    srand(42);
    float *w = malloc((size_t)d * sizeof(float));
    for (int i = 0; i < d; i += 2) {
        float u1 = (float)(rand()+1) / (float)RAND_MAX;
        float u2 = (float)(rand()+1) / (float)RAND_MAX;
        float r = sqrtf(-2.0f * logf(u1));
        w[i]   = r * cosf(2.0f * (float)M_PI * u2) * 0.1f;
        w[i+1] = r * sinf(2.0f * (float)M_PI * u2) * 0.1f;
    }

    /* Quantize → dequantize roundtrip */
    wubu_nf4_block *blocks = malloc((size_t)n_blocks * sizeof(wubu_nf4_block));
    wubu_nf4_quantize(w, blocks, d);

    float *w_dq = malloc((size_t)d * sizeof(float));
    wubu_nf4_dequantize(blocks, w_dq, d);

    /* Cosine similarity */
    float dot=0, n1=0, n2=0, maxd=0;
    for (int i = 0; i < d; i++) {
        dot += w[i] * w_dq[i];
        n1 += w[i] * w[i];
        n2 += w_dq[i] * w_dq[i];
        float diff = fabsf(w[i] - w_dq[i]);
        if (diff > maxd) maxd = diff;
    }
    float cs = dot / (sqrtf(n1) * sqrtf(n2) + 1e-10f);

    printf("Vector dim:           %d (%d blocks)\n", d, n_blocks);
    printf("Storage:              %d bytes (vs %d F32 = %.1fx compression)\n",
           n_blocks * (int)sizeof(wubu_nf4_block), d*4,
           (float)(d*4) / (float)(n_blocks * sizeof(wubu_nf4_block)));
    printf("NF4 roundtrip cosine: %.6f\n", cs);
    printf("NF4 max abs diff:     %.6f\n", maxd);

    /* Test fused dot product */
    float *q = malloc((size_t)d * sizeof(float));
    for (int i = 0; i < d; i++) q[i] = (float)sin(i*0.01) * 0.5f;

    float dot_f32 = 0;
    for (int i = 0; i < d; i++) dot_f32 += q[i] * w[i];

    float dot_nf4 = wubu_nf4_dequant_dot(q, blocks, d);
    float dot_err = fabsf(dot_f32 - dot_nf4) / (fabsf(dot_f32) + 1e-10f);

    printf("\nFused dequant+dot:\n");
    printf("  F32 dot:         %.4f\n", dot_f32);
    printf("  NF4 dot:         %.4f\n", dot_nf4);
    printf("  Relative error:  %.6f\n", dot_err);

    int err = 0;
    if (cs < 0.99f) { printf("FAIL: cosine %.4f < 0.99\n", cs); err++; }
    if (dot_err > 0.05f) { printf("FAIL: dot relative error %.4f > 0.05\n", dot_err); err++; }

    printf("\n%s: NF4 roundtrip cosine %.4f, dot rel error %.4f\n",
           err ? "FAIL" : "PASS", cs, dot_err);
    printf("=== %d errors ===\n", err);

    free(w); free(blocks); free(w_dq); free(q);
    return err;
}
