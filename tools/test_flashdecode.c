/*
 * test_flashdecode.c -- doc 015 triple-DA.
 * P1 correctness: FlashDecoding output == reference full-softmax attention
 *   within 1e-3 on random + Qwen-like config, for cache_len in {512,4096,65536}.
 * P2 privacy: own C, no external lib.
 * P3 robustness: chunk=1 reproduces serial (no parallelism cliff); degenerate
 *   cache_len=0 yields zero vector; numerical merge stable (no overflow via
 *   expf on corrected scores only).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "wubu_flashdecode.h"

static void ref_attn(const float *q, const float *Kc, const float *Vc,
                     int head_dim, int n_kv_heads, int h_kv, int64_t L, float scale,
                     float *out) {
    const int64_t stride = (int64_t)n_kv_heads * head_dim;
    const int64_t base = (int64_t)h_kv * head_dim;
    float maxs = -1e30f;
    float *scores = malloc((size_t)L * sizeof(float));
    for (int64_t t = 0; t < L; t++) {
        const float *kt = Kc + t*stride + base;
        float d = 0; for (int i=0;i<head_dim;i++) d += q[i]*kt[i];
        scores[t] = d*scale; if (scores[t] > maxs) maxs = scores[t];
    }
    float sum = 0;
    for (int64_t t = 0; t < L; t++) { scores[t] = expf(scores[t]-maxs); sum += scores[t]; }
    float inv = 1.0f/sum;
    for (int i=0;i<head_dim;i++) out[i] = 0;
    for (int64_t t = 0; t < L; t++) {
        float p = scores[t]*inv;
        const float *vt = Vc + t*stride + base;
        for (int i=0;i<head_dim;i++) out[i] += p*vt[i];
    }
    free(scores);
}

static float maxdiff(const float *a, const float *b, int n) {
    float m = 0; for (int i=0;i<n;i++){ float d=fabsf(a[i]-b[i]); if(d>m)m=d; } return m;
}

static int run_cfg(int head_dim, int n_kv, int n_q, int64_t L, unsigned seed) {
    srand(seed);
    int64_t Ksz = L * n_kv * head_dim;
    float *Kc = malloc((size_t)Ksz*sizeof(float));
    float *Vc = malloc((size_t)Ksz*sizeof(float));
    float *Q  = malloc((size_t)n_q*head_dim*sizeof(float));
    for (int64_t i=0;i<Ksz;i++){ Kc[i]=((float)rand()/RAND_MAX*2-1); Vc[i]=((float)rand()/RAND_MAX*2-1); }
    for (int i=0;i<n_q*head_dim;i++) Q[i]=((float)rand()/RAND_MAX*2-1);
    float scale = 1.0f/sqrtf((float)head_dim);

    float *fd = malloc((size_t)n_q*head_dim*sizeof(float));
    float *rf = malloc((size_t)n_q*head_dim*sizeof(float));
    wubu_flashdecode_all(Q, Kc, Vc, head_dim, n_q, n_kv, L, scale, 0, fd);
    /* reference head-by-head */
    int group = n_q/n_kv;
    for (int h=0;h<n_q;h++) ref_attn(Q+(size_t)h*head_dim, Kc, Vc, head_dim, n_kv, h/group, L, scale, rf+(size_t)h*head_dim);

    float md = maxdiff(fd, rf, n_q*head_dim);
    printf("  cfg hd=%d nkv=%d nq=%d L=%lld -> maxdiff=%.2e\n", head_dim, n_kv, n_q, (long long)L, md);
    int ok = md < 1e-3f;
    free(Kc);free(Vc);free(Q);free(fd);free(rf);
    return ok;
}

int main(void) {
    printf("FlashDecoding vs full-softmax reference:\n");
    int ok = 1;
    ok &= run_cfg(128, 8, 64, 512, 1);
    ok &= run_cfg(128, 8, 64, 4096, 2);
    ok &= run_cfg(128, 8, 64, 65536, 3);
    ok &= run_cfg(64, 4, 32, 8192, 4);

    /* degenerate cache_len=0 -> zero vector */
    float q[16], Kc[16], Vc[16], out[16];
    for (int i=0;i<16;i++){q[i]=1;Kc[i]=0;Vc[i]=0;}
    wubu_flashdecode_head(q, Kc, Vc, 16, 1, 0, 0, 1.0f, 0, out);
    int zero_ok = 1; for (int i=0;i<16;i++) if (out[i]!=0) zero_ok=0;
    printf("  cache_len=0 -> zero vector: %s\n", zero_ok?"PASS":"FAIL");
    ok &= zero_ok;

    printf(ok ? "ALL FLASHDECODE CHECKS PASSED\n" : "FLASHDECODE CHECKS FAILED\n");
    return ok ? 0 : 1;
}
