/* test_cross_attn.c — Cross-attention (multimodal fusion) correctness test
 * Validates F32 and Q8 cross-attention between two modalities.
 */
#include "wubu_cross_attn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float vec_dot(float *a, float *b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i]*b[i];
    return s;
}
static float vec_norm(float *v, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += v[i]*v[i];
    return sqrtf(s);
}

int main(void) {
    printf("=== Cross-Attention (Multimodal Fusion) Test ===\n\n");
    int n_q=8, n_kv=4, hd=128, max_kv=512, enc_len=256;

    wubu_cross_attn_ctx_t *ctx = wubu_cross_attn_init(n_q, n_kv, hd, max_kv);
    if (!ctx) { printf("init FAIL\n"); return 1; }

    /* Simulate encoder (vision) output: K/V with position structure */
    float *k_enc = malloc((size_t)enc_len * n_kv * hd * sizeof(float));
    float *v_enc = malloc((size_t)enc_len * n_kv * hd * sizeof(float));
    for (int t = 0; t < enc_len; t++) {
        for (int h = 0; h < n_kv; h++) {
            for (int i = 0; i < hd; i++) {
                k_enc[(t*n_kv+h)*hd+i] = (float)sin(t*0.03+i*0.12+h*0.4)*2.0f;
                v_enc[(t*n_kv+h)*hd+i] = (float)cos(t*0.04+i*0.10+h*0.2)*2.0f;
            }
        }
    }

    /* Simulate decoder (text) query */
    float *q = malloc((size_t)n_q * hd * sizeof(float));
    for (int i = 0; i < n_q; i++)
        for (int j = 0; j < hd; j++)
            q[i*hd+j] = (float)sin(i*0.2+j*0.08)*1.5f;

    /* F32 cross-attention */
    wubu_cross_attn_store_kv(ctx, k_enc, v_enc, enc_len);
    float *out_f32 = calloc((size_t)n_q * hd, sizeof(float));
    wubu_cross_attn_decode(ctx, q, out_f32, 4);

    /* Q8 cross-attention — same encoder output */
    wubu_cross_attn_store_kv_q8(ctx, k_enc, v_enc, enc_len);
    float *out_q8 = calloc((size_t)n_q * hd, sizeof(float));
    wubu_cross_attn_decode_q8(ctx, q, out_q8, 1);

    /* Compare F32 vs Q8 */
    float maxd = 0, dot = 0, n1 = 0, n2 = 0;
    for (int i = 0; i < n_q*hd; i++) {
        float d = fabsf(out_f32[i] - out_q8[i]);
        if (d > maxd) maxd = d;
        dot += out_f32[i] * out_q8[i];
        n1 += out_f32[i] * out_f32[i];
        n2 += out_q8[i] * out_q8[i];
    }
    float cs = dot / (sqrtf(n1) * sqrtf(n2) + 1e-10f);
    float norm = vec_norm(out_f32, n_q*hd);

    printf("Encoder: %d tokens, %d KV heads, %d dim\n", enc_len, n_kv, hd);
    printf("Decoder: %d query heads, GQA group=%d\n", n_q, n_q/n_kv);
    printf("F32 output norm:       %.4f\n", norm);
    printf("Q8 output norm:        %.4f\n", vec_norm(out_q8, n_q*hd));
    printf("F32 vs Q8 max_diff:    %.6f\n", maxd);
    printf("F32 vs Q8 cosine:      %.6f\n", cs);
    printf("Q8 compression:       %.1fx vs F32\n",
           (double)(hd*4)/(double)(((hd+31)/32)*36));

    int err = (cs < 0.999f) ? 1 : 0;
    printf("\n%s: cross-attention F32 vs Q8 cosine %.4f %s 0.999\n",
           err ? "FAIL" : "PASS", cs, err ? "<" : ">=");
    printf("=== %d errors ===\n", err);

    free(k_enc); free(v_enc); free(q);
    free(out_f32); free(out_q8);
    wubu_cross_attn_free(ctx);
    return err;
}
