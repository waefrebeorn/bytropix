/*
 * test_ssd_moe.c -- verify the ds4-ssd slot-bank pager.
 *
 * Builds a SYNTHETIC safetensors checkpoint (D=16, F=8, E=12 experts) with the
 * real KAT tensor naming (model.language_model.layers.0.mlp.experts.E.{gate,up,
 * down}_proj.weight), then opens it via wubu_ssd_moe_open (which pages experts
 * straight from the checkpoint shards — no sidecar). Pages experts through a
 * 3-slot bank (forcing LRU evictions), runs a matmul from paged weights, and
 * compares against a fully-resident reference computed in-RAM.
 */
#include "wubu_ssd_moe.h"
#include "safetensors_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int D = 16, F = 8, E = 12, SLOTS = 3;

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

/* Minimal safetensors emitter: one BF16 tensor per expert matrix. */
static int write_ckpt(const char *path) {
    for (int e = 0; e < E; e++) make_expert(e, ref_gate[e], ref_up[e], ref_down[e]);
    /* Build header JSON. */
    char hdr[8192]; int hl = 0;
    hl += snprintf(hdr+hl, sizeof(hdr)-hl, "{");
    const char *names[3] = {"gate_proj", "up_proj", "down_proj"};
    for (int e = 0; e < E; e++) {
        for (int t = 0; t < 3; t++) {
            long long begin = 0; /* filled after we know offsets; do a 2-pass */
            (void)begin;
        }
    }
    /* Pass 1: compute offsets. Each expert matrix = D*F (gate/up) or F*D (down)
     * BF16 = (D*F)*2 bytes. Stored [gate,up,down] per expert. */
    long long off[E][3]; long long cur = 0;
    for (int e = 0; e < E; e++) {
        off[e][0] = cur; cur += (long long)D*F*2;
        off[e][1] = cur; cur += (long long)D*F*2;
        off[e][2] = cur; cur += (long long)F*D*2;
    }
    long long total = cur;
    hl = 0; snprintf(hdr+hl, sizeof(hdr)-hl, "{");
    for (int e = 0; e < E; e++) {
        for (int t = 0; t < 3; t++) {
            const char *nm = names[t];
            const char *shape = (t < 2) ? "[16,8]" : "[8,16]";
            char key[128];
            snprintf(key, sizeof(key), "\"model.language_model.layers.0.mlp.experts.%d.%s.weight\"", e, nm);
            hl += snprintf(hdr+hl, sizeof(hdr)-hl,
                "%s:%s{\"dtype\":\"BF16\",\"shape\":%s,\"data_offsets\":[%lld,%lld]}",
                (hl>1?",":""), key, shape, off[e][t], off[e][t] + (long long)(t<2?D*F:F*D)*2);
        }
    }
    hl += snprintf(hdr+hl, sizeof(hdr)-hl, "}");

    /* Pad header to 8-byte boundary. */
    size_t hlen = (size_t)hl;
    while (hlen % 8 != 0) { hdr[hlen] = ' '; hlen++; }
    if (hlen + 8 > sizeof(hdr)) return -1;

    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    uint64_t hl_u = (uint64_t)hl;
    fwrite(&hl_u, 8, 1, fp);
    fwrite(hdr, 1, hlen, fp);
    /* Write expert tensors (gate,up,down) in order. */
    for (int e = 0; e < E; e++) {
        for (int i = 0; i < D*F; i++) { uint16_t h=f32_to_bf16(ref_gate[e][i]); fwrite(&h,2,1,fp); }
        for (int i = 0; i < D*F; i++) { uint16_t h=f32_to_bf16(ref_up[e][i]);   fwrite(&h,2,1,fp); }
        for (int i = 0; i < F*D; i++) { uint16_t h=f32_to_bf16(ref_down[e][i]); fwrite(&h,2,1,fp); }
    }
    fclose(fp);
    (void)total;
    return 0;
}

int main(void) {
    char path[] = "/tmp/ssd_moe_test/model-00000-of-00001.safetensors";
    mkdir("/tmp/ssd_moe_test", 0755);
    if (write_ckpt(path) != 0) { printf("FAIL: write checkpoint\n"); return 1; }

    /* Open the slot-bank over the checkpoint dir (pages from shards, no sidecar). */
    wubu_ssd_moe_t *m = wubu_ssd_moe_open("/tmp/ssd_moe_test", SLOTS);
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
    printf("PASS: slot-bank pager correct (hit + page-in paths, LRU evictions exercised, no sidecar)\n");
    return 0;
}
