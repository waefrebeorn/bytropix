/*
 * test_gemv_int4.c -- B03: int4 weight GEMV vs fp32 oracle (triple-DA).
 * Pass 1 correctness: cosine(int4, fp32) > 0.99 on random + real Qwen.
 * Pass 3 robustness: autotuner precedence int4 > int8 > fp32 by shape.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_gemm.h"
#include "wubu_gemv_tune.h"
#include "wubu_safetensors_shard.h"

static float cosine(const float *a, const float *b, int n) {
    double d = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return (float)(d / (sqrt(na)*sqrt(nb) + 1e-12));
}

static void scalar_gemv(const float *w, const float *x, float *y, int M, int K) {
    for (int m = 0; m < M; m++) {
        double s = 0; const float *wr = w + (size_t)m*K;
        for (int k = 0; k < K; k++) s += wr[k]*x[k];
        y[m] = (float)s;
    }
}

static int failures = 0;
#define CHECK(c,msg) do { if(!(c)){ printf("  FAIL: %s\n", msg); failures++; } else { printf("  PASS: %s\n", msg); } } while(0)

int main(void) {
    srand(1234);
    int shapes[][2] = { {4096,2048}, {8192,4096}, {512,512} };

    for (int s = 0; s < 3; s++) {
        int M = shapes[s][0], K = shapes[s][1];
        float *w = malloc((size_t)M*K*sizeof(float));
        float *x = malloc(K*sizeof(float));
        float *yf = malloc(M*sizeof(float));
        float *y4 = malloc(M*sizeof(float));
        float *yi = malloc(M*sizeof(float));
        for (int i = 0; i < M*K; i++) w[i] = ((float)rand()/RAND_MAX*2-1) * 3.0f;
        for (int i = 0; i < K; i++) x[i] = ((float)rand()/RAND_MAX*2-1);
        scalar_gemv(w, x, yf, M, K);

        /* int4 path */
        int8_t *q4 = malloc((size_t)M*((K+1)/2));
        float *sc = malloc(M*sizeof(float));
        wubu_gemv_quantize_i4(w, q4, sc, M, K);
        wubu_gemv_i4(q4, sc, x, y4, M, K);

        /* int8 path */
        int8_t *qi = malloc((size_t)M*K);
        float *si = malloc(M*sizeof(float));
        wubu_gemv_quantize_i8(w, qi, si, M, K);
        wubu_gemv_i8(qi, si, x, yi, M, K);

        printf("shape M=%d K=%d:\n", M, K);
        CHECK(cosine(y4, yf, M) > 0.99f, "int4 cosine > 0.99");
        CHECK(cosine(yi, yf, M) > 0.995f, "int8 cosine > 0.995");
        for (int i = 0; i < M; i++) if (!isfinite(y4[i]) || !isfinite(yi[i])) { CHECK(0,"finite"); break; }

        wubu_gemv_tile_t t = wubu_gemv_autotune(M, K, 0.0);
        const char *tn = wubu_gemv_tile_name(&t);
        printf("  tune M=%d: %s\n", M, tn);

        free(w); free(x); free(yf); free(y4); free(yi);
        free(q4); free(sc); free(qi); free(si);
    }

    /* autotune precedence by shape */
    wubu_gemv_tile_t t_big = wubu_gemv_autotune(8192, 4096, 0.0);
    wubu_gemv_tile_t t_mid = wubu_gemv_autotune(512, 512, 0.0);
    wubu_gemv_tile_t t_tiny = wubu_gemv_autotune(16, 16, 0.0);
    CHECK(t_big.use_int4 == 1, "big shape -> int4");
    CHECK(t_mid.use_int8 == 1 && t_mid.use_int4 == 0, "mid shape -> int8 only");
    CHECK(t_tiny.use_int8 == 0 && t_tiny.use_int4 == 0, "tiny shape -> fp32");

    /* real Qwen gate_proj (5120x5120 bf16 -> load fp32) */
    {
        const char *path = "/home/wubu/models/Qwen3.6-27B";
        wubu_shard_ctx_t *st = wubu_shard_open(path);
        if (st) {
            const char *name = "model.layers.0.self_attn.q_proj.weight";
            int64_t ne = 0;
            const float *w = wubu_shard_data_f32(st, name, &ne);
            if (w && ne == 5120LL*5120LL) {
                int M = 5120, K = 5120;
                float *x = malloc(K*sizeof(float));
                float *yf = malloc(M*sizeof(float));
                float *y4 = malloc(M*sizeof(float));
                for (int i = 0; i < K; i++) x[i] = ((float)rand()/RAND_MAX*2-1);
                scalar_gemv(w, x, yf, M, K);
                int8_t *q4 = malloc((size_t)M*((K+1)/2));
                float *sc = malloc(M*sizeof(float));
                wubu_gemv_quantize_i4(w, q4, sc, M, K);
                wubu_gemv_i4(q4, sc, x, y4, M, K);
                printf("qwen gate_proj int4 cosine=%.6f\n", cosine(y4, yf, M));
                CHECK(cosine(y4, yf, M) > 0.99f, "qwen int4 cosine > 0.99");
                free(x); free(yf); free(y4); free(q4); free(sc);
            } else { printf("  (skip real-weight: shard/weight unavailable)\n"); }
            wubu_shard_close(st);
        } else { printf("  (skip real-weight: Qwen model dir unavailable)\n"); }
    }

    printf("\n%s\n", failures ? "SOME CHECKS FAILED" : "ALL GEMV-INT4 CHECKS PASSED");
    return failures ? 1 : 0;
}
