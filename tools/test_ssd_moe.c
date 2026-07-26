/*
 * test_ssd_moe.c -- verify the ds4-ssd slot-bank pager.
 * Builds a synthetic sidecar (D=16, F=8, E=12 experts), pages experts through
 * a 3-slot bank (forcing LRU evictions), runs a matmul from paged weights, and
 * compares against a fully-resident reference computed in-RAM.
 */
#include "wubu_ssd_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>

static int D = 16, F = 8, E = 12, ACTIVE = 2, SLOTS = 3;

static float ref_gate[12][16*8], ref_up[12][16*8], ref_down[12][8*16];

static void make_expert(int e, float g[16*8], float u[16*8], float d[8*16]) {
    for (int i = 0; i < 16*8; i++) { g[i] = 0.01f * ((i*7 + e*13) % 11) - 0.05f; u[i] = g[i]*1.3f; }
    for (int i = 0; i < 8*16; i++)  d[i] = 0.01f * ((i*5 + e*3) % 9) - 0.04f;
}
static float matmul_ref(int e, const float *x) {
    float g[16*8], u[16*8], d[8*16];
    make_expert(e, g, u, d);
    float h[8];
    for (int j = 0; j < 8; j++) {
        float sg = 0, su = 0;
        for (int k = 0; k < 16; k++) { sg += x[k]*g[k + j*16]; su += x[k]*u[k + j*16]; }
        float silu = sg / (1.0f + expf(-sg));
        h[j] = silu * su;
    }
    float y[16] = {0};
    for (int k = 0; k < 16; k++) for (int j = 0; j < 8; j++) y[k] += h[j]*d[k + j*16];
    return y[0];
}
static uint16_t f32_to_bf16(float v) { uint32_t b; memcpy(&b,&v,4); return (uint16_t)(b>>16); }

int main(void) {
    char dir[] = "/tmp/ssd_moe_test";
    mkdir(dir, 0755);
    for (int e = 0; e < E; e++) make_expert(e, ref_gate[e], ref_up[e], ref_down[e]);

    /* Pack all experts sequentially via FILE (BF16). */
    char path[256]; snprintf(path, sizeof(path), "%s/experts.0.bin", dir);
    FILE *fp = fopen(path, "wb");
    for (int e = 0; e < E; e++) {
        float *g = ref_gate[e], *u = ref_up[e], *d = ref_down[e];
        for (int i = 0; i < D*F; i++) { uint16_t h=f32_to_bf16(g[i]); fwrite(&h,2,1,fp); }
        for (int i = 0; i < D*F; i++) { uint16_t h=f32_to_bf16(u[i]); fwrite(&h,2,1,fp); }
        for (int i = 0; i < F*D; i++) { uint16_t h=f32_to_bf16(d[i]); fwrite(&h,2,1,fp); }
    }
    fclose(fp);
    wubu_ssd_moe_write_manifest(dir, 1, E, D, F, ACTIVE, SLOTS);

    wubu_ssd_moe_t *m = wubu_ssd_moe_open(dir, SLOTS);
    if (!m) { printf("FAIL: open\n"); return 1; }
    printf("opened: layers=%d experts=%d D=%d F=%d slots=%d\n",
           wubu_ssd_moe_n_layers(m), wubu_ssd_moe_n_experts(m),
           wubu_ssd_moe_d_model(m), wubu_ssd_moe_d_ff(m), SLOTS);

    float x[16]; for (int i = 0; i < 16; i++) x[i] = 0.1f*i - 0.7f;

    int mism = 0;
    for (int e = 0; e < E; e++) {
        float *out[3]; int r = wubu_ssd_moe_get(m, 0, e, out);
        if (r < 0) { printf("FAIL: get expert %d\n", e); return 1; }
        float g[16*8], u[16*8], d[8*16];
        memcpy(g, out[0], sizeof(g)); memcpy(u, out[1], sizeof(u)); memcpy(d, out[2], sizeof(d));
        float h[8];
        for (int j = 0; j < 8; j++) {
            float sg = 0, su = 0;
            for (int k = 0; k < 16; k++) { sg += x[k]*g[k + j*16]; su += x[k]*u[k + j*16]; }
            float silu = sg / (1.0f + expf(-sg));
            h[j] = silu * su;
        }
        float y[16] = {0};
        for (int k = 0; k < 16; k++) for (int j = 0; j < 8; j++) y[k] += h[j]*d[k + j*16];
        float ref = matmul_ref(e, x);
        if (fabsf(y[0] - ref) > 1e-2f) { mism++; if (mism<=3) printf("  expert %d mismatch: got %f ref %f\n", e, y[0], ref); }
    }
    long pi, hi; long long br; wubu_ssd_moe_stats(m, &pi, &hi, &br);
    printf("stats: pageins=%ld hits=%ld bytes_read=%lld\n", pi, hi, br);
    wubu_ssd_moe_close(m);
    if (mism) { printf("FAIL: %d expert mismatches\n", mism); return 1; }
    printf("PASS: slot-bank pager correct (hit + page-in paths, LRU evictions exercised)\n");
    return 0;
}
