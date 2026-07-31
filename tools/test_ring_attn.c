#include "wubu_ring_attn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    printf("=== Ring Attention Test ===\n\n");
    int n_heads=2, hd=64, ctx_len=256, n_chunks=4;

    wubu_ring_attn_ctx_t *ctx = wubu_ring_attn_init(n_heads, hd, ctx_len, n_chunks);
    if (!ctx) { printf("FAIL: init\n"); return 1; }

    size_t total = (size_t)ctx_len * n_heads * hd;
    float *k = malloc(total * sizeof(float));
    float *v = malloc(total * sizeof(float));
    float *q = malloc(total * sizeof(float));
    float *out = calloc(total, sizeof(float));

    for (size_t i = 0; i < total; i++) {
        k[i] = sinf(i * 0.01f) * 2.0f;
        v[i] = cosf(i * 0.01f) * 2.0f;
        q[i] = sinf(i * 0.01f) * 1.5f;
    }

    int rc = wubu_ring_attn_forward(ctx, q, k, v, ctx_len, n_chunks, out, 2);
    if (rc != 0) { printf("FAIL: forward returned %d\n", rc); return 1; }

    float norm = 0.0f;
    for (size_t i = 0; i < total; i++) norm += out[i] * out[i];
    norm = sqrtf(norm);

    printf("Context: %d tokens, %d chunks, %d heads, dim=%d\n", ctx_len, n_chunks, n_heads, hd);
    printf("Output norm: %.4f\n", norm);

    int err = (norm < 0.001f) ? 1 : 0;
    printf("%s: ring attention norm %.4f\n", err ? "FAIL" : "PASS", norm);

    free(k); free(v); free(q); free(out);
    wubu_ring_attn_free(ctx);
    printf("=== %d errors ===\n", err);
    return err;
}
