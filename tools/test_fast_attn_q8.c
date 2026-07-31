/* test_fast_attn_q8.c — Q8 KV cache fast decode benchmark + correctness */
#include "wubu_fast_attn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct { float d; int8_t qs[32]; } __attribute__((packed)) q8_block;

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void quant_row_q8(const float *src, q8_block *dst, int n) {
    int nb = (n + 31) / 32;
    for (int b = 0; b < nb; b++) {
        int off = b * 32;
        int cnt = (off + 32 <= n) ? 32 : (n - off);
        float amax = 0.0f;
        for (int i = 0; i < cnt; i++) { float a = fabsf(src[off+i]); if (a > amax) amax = a; }
        float sc = (amax > 1e-8f) ? amax / 127.0f : 1e-8f;
        dst[b].d = sc;
        for (int i = 0; i < cnt; i++) {
            int v = (int)roundf(src[off+i] / sc);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            dst[b].qs[i] = (int8_t)v;
        }
        for (int i = cnt; i < 32; i++) dst[b].qs[i] = 0;
    }
}

int main(void) {
    int n_q = 16, n_kv = 2, hd = 128, n_rot = 64;
    wubu_fast_attn_ctx_t *ctx = wubu_fast_attn_init(
            n_q, n_kv, hd, 512*1024, n_rot, 10000000.0f, 0.25f);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 1; }

    int bph = (hd + 31) / 32;
    int kvhb = bph * (int)sizeof(q8_block);
    int sizes[] = {4096, 16384, 65536, 262144};
    int nsz = 4;
    int errors = 0;

    for (int si = 0; si < nsz; si++) {
        int cl = sizes[si];
        printf("\n=== Q8 Context: %d tokens ===\n", cl);

        float *q = malloc((size_t)n_q*hd*sizeof(float));
        float *kf = malloc((size_t)cl*n_kv*hd*sizeof(float));
        float *vf = malloc((size_t)cl*n_kv*hd*sizeof(float));
        float *oq8 = malloc((size_t)n_q*hd*sizeof(float));
        float *of32 = malloc((size_t)n_q*hd*sizeof(float));
        float *otiled = malloc((size_t)n_q*hd*sizeof(float));
        q8_block *kq8 = malloc((size_t)cl*n_kv*kvhb);
        q8_block *vq8 = malloc((size_t)cl*n_kv*kvhb);
        if (!q||!kf||!vf||!oq8||!of32||!otiled||!kq8||!vq8) { fprintf(stderr,"OOM\n"); break; }

        for (int i = 0; i < n_q*hd; i++) q[i] = (float)((i*7+13)%17-8)*0.01f;
        for (int i = 0; i < cl*n_kv*hd; i++) {
            kf[i] = (float)((i*3+1)%19-9)*0.01f;
            vf[i] = (float)((i*5+7)%23-11)*0.01f;
        }
        for (int t = 0; t < cl; t++)
            for (int g = 0; g < n_kv; g++) {
                quant_row_q8(kf+(size_t)t*n_kv*hd+g*hd, kq8+(size_t)t*n_kv*bph+g*bph, hd);
                quant_row_q8(vf+(size_t)t*n_kv*hd+g*hd, vq8+(size_t)t*n_kv*bph+g*bph, hd);
            }

        float *kn = malloc((size_t)n_kv*hd*sizeof(float));
        memcpy(kn, kf+(size_t)(cl-1)*n_kv*hd, (size_t)n_kv*hd*sizeof(float));
        wubu_fast_attn_rope(ctx, q, kn, cl-1);
        free(kn);

        double t0 = now_ms();
        wubu_fast_attn_decode(ctx, q, kf, vf, cl, of32, 6);
        double tf = now_ms() - t0;

        double t1 = now_ms();
        wubu_fast_attn_decode_q8(ctx, q, kq8, vq8, cl, oq8, 6);
        double tq8 = now_ms() - t1;

        double t2 = now_ms();
        wubu_fast_attn_decode_q8_tiled(ctx, q, kq8, vq8, cl, otiled, 6, 0);
        double tt = now_ms() - t2;

        /* Correctness: Q8 vs F32 */
        float md = 0.0f;
        for (int i = 0; i < n_q*hd; i++) { float d = fabsf(oq8[i]-of32[i]); if (d > md) md = d; }
        printf("[Q8]    vs F32:  max_diff=%.2e %s\n", (double)md, md < 0.05f ? "PASS" : "FAIL");
        if (md > 0.05f) errors++;

        /* Tiled vs untiled Q8 */
        float td = 0.0f;
        for (int i = 0; i < n_q*hd; i++) { float d = fabsf(oq8[i]-otiled[i]); if (d > td) td = d; }
        printf("[tiled] vs Q8:   max_diff=%.2e %s\n", (double)td, td < 1e-5f ? "PASS" : "FAIL");
        if (td > 1e-5f) errors++;

        size_t fb = (size_t)cl*n_kv*hd*4*2;
        size_t qb = (size_t)cl*n_kv*kvhb*2;
        printf("[timing] F32=%.1fms Q8=%.1fms tiled=%.1fms | F32→Q8=%.2fx Q8→tiled=%.2fx\n",
               tf, tq8, tt, tf/tq8, tq8/tt);
        printf("[bandwidth] F32=%.1f MB Q8=%.1f MB (%.0f%% less data)\n",
               (double)fb/1e6, (double)qb/1e6, 100.0*(1.0-(double)qb/fb));

        free(q); free(kf); free(vf); free(oq8); free(of32); free(otiled);
        free(kq8); free(vq8);
    }

    printf("\n=== Summary: %d errors ===\n", errors);
    wubu_fast_attn_free(ctx);
    return errors;
}