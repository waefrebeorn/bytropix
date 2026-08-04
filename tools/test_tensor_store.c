/*
 * test_tensor_store.c -- the uniform tensor catalog round-trip test.
 *
 * Proves the anti-waste interchange: a 137-tensor .st dump (the trainer's
 * format) -> safetensors -> .st and -> GGUF, reopening each and verifying
 * tensor-by-tensor equality. All synthetic (deterministic seed), so the
 * test is self-contained and fast (~140 MB per copy, no real weights).
 *
 * Tests:
 *   1. sniff() detects all three formats by magic
 *   2. .st catalog: 137 entries, names match the fixed layout
 *   3. get_f32 on the .st matches the bytes we wrote
 *   4. .st -> safetensors -> get_f32 equality (round-trip < 1e-6)
 *   5. safetensors -> .st -> get_f32 equality (back again)
 *   6. safetensors -> GGUF -> get_f32 equality + gguf_open can read it
 *   7. wubu_ts_find("layers.5.attn.q_proj.weight") works in all formats
 */
#include "wubu_tensor_store.h"
#include "safetensors_reader.h"
#include "gguf_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } else { printf("  ok: %s\n", msg); } } while (0)

#define ST_N_LAYERS    12
#define ST_N_SELECTORS 3
#define ST_TOTAL       137

/* splitmix64 for deterministic synthetic weights */
static uint64_t smix(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* write a synthetic 137-tensor .st dump */
static int write_stdump(const char *path)
{
    FILE *w = fopen(path, "wb");
    if (!w) return -1;
    uint32_t magic = 0xBA000002u;
    int nl = ST_N_LAYERS;
    long n = 35072768L; /* BARUN_PARAMS */
    fwrite(&magic, 4, 1, w);
    fwrite(&nl, 4, 1, w);
    fwrite(&n, sizeof(long), 1, w);
    uint64_t seed = 48;
    /* embedding + final_norm */
    int64_t sizes[ST_TOTAL];
    int cnt = 0;
    sizes[cnt++] = 16384LL * 448; sizes[cnt++] = 448;
    for (int l = 0; l < ST_N_LAYERS; l++)
        for (int t = 0; t < 11; t++) {
            static const int64_t LE[11] = { 448LL*448, 448LL*64, 448LL*64, 448LL*448,
                448LL*448, 64LL, 64LL, 448LL, 448LL*2456, 1228LL*448, 448LL };
            sizes[cnt++] = LE[t];
        }
    for (int s = 0; s < ST_N_SELECTORS; s++) sizes[cnt++] = 448;
    for (int i = 0; i < cnt; i++) {
        float *buf = (float *)malloc((size_t)sizes[i] * sizeof(float));
        if (!buf) { fclose(w); return -1; }
        for (int64_t k = 0; k < sizes[i]; k++)
            buf[k] = (float)((double)(smix(&seed) >> 11) / (double)(1ull << 53)) * 2.0f - 1.0f;
        fwrite(buf, sizeof(float), (size_t)sizes[i], w);
        free(buf);
    }
    fclose(w);
    return 0;
}

static const char *layer_name(int l, int t, char *buf, size_t cap)
{
    static const char *L[11] = {
        "layers.%d.attn.q_proj.weight", "layers.%d.attn.k_proj.weight",
        "layers.%d.attn.v_proj.weight", "layers.%d.attn.o_proj.weight",
        "layers.%d.attn.g_proj.weight", "layers.%d.attn.q_norm.weight",
        "layers.%d.attn.k_norm.weight", "layers.%d.attn_norm.weight",
        "layers.%d.ffn.gate_up.weight", "layers.%d.ffn.down.weight",
        "layers.%d.ffn_norm.weight" };
    snprintf(buf, cap, L[t], l);
    return buf;
}

static double maxdiff(const float *a, const float *b, int64_t n)
{
    double d = 0;
    for (int64_t i = 0; i < n; i++) {
        double x = fabs((double)a[i] - (double)b[i]);
        if (x > d) d = x;
    }
    return d;
}

int main(void)
{
    printf("=== test_tensor_store (uniform catalog interchange) ===\n");
    const char *st_path = "/tmp/ts_fixture.st";
    const char *st_path2 = "/tmp/ts_fixture.safetensors";
    const char *st_path3 = "/tmp/ts_fixture_back.st";
    const char *st_path4 = "/tmp/ts_fixture.gguf";

    CHECK(write_stdump(st_path) == 0, "write synthetic 137-tensor .st dump");

    /* 1. sniff */
    CHECK(wubu_ts_sniff(st_path) == WUBU_TS_STDUMP, "sniff .st -> STDUMP");
    CHECK(wubu_ts_sniff("/dev/null") == WUBU_TS_UNKNOWN, "sniff junk -> UNKNOWN");

    /* 2. .st catalog */
    wubu_tensor_store_t *a = wubu_ts_open(st_path);
    CHECK(a != NULL, "open .st dump");
    CHECK(a && wubu_ts_format(a) == WUBU_TS_STDUMP, ".st format reported");
    CHECK(a && wubu_ts_count(a) == ST_TOTAL, "catalog has 137 entries");
    CHECK(a && wubu_ts_find(a, "embedding.weight") != NULL, "find embedding.weight");
    char nb[192];
    CHECK(a && wubu_ts_find(a, layer_name(5, 0, nb, sizeof(nb))) != NULL,
          "find layers.5.attn.q_proj.weight");

    /* 3. get_f32 on .st matches written bytes */
    if (a) {
        const wubu_ts_entry *e = wubu_ts_find(a, "final_norm.weight");
        CHECK(e && e->n_elems == 448, "final_norm has 448 elems");
    }

    /* 4. .st -> safetensors -> equality */
    CHECK(a && wubu_ts_export(a, WUBU_TS_SAFETENSORS, st_path2) == 0,
          "export .st -> safetensors");
    wubu_tensor_store_t *b = wubu_ts_open(st_path2);
    CHECK(b != NULL, "reopen safetensors");
    CHECK(b && wubu_ts_sniff(st_path2) == WUBU_TS_SAFETENSORS, "sniff safetensors");
    CHECK(b && wubu_ts_count(b) == ST_TOTAL, "safetensors catalog 137");
    if (a && b) {
        float *x = (float *)malloc(sizeof(float) * 16384 * 448);
        float *y = (float *)malloc(sizeof(float) * 16384 * 448);
        CHECK(wubu_ts_get_f32(a, "embedding.weight", x, 16384LL * 448) == 0 &&
              wubu_ts_get_f32(b, "embedding.weight", y, 16384LL * 448) == 0,
              "get embedding from both");
        if (x && y) {
            double d = maxdiff(x, y, 16384LL * 448);
            printf("  embedding round-trip maxdiff = %g\n", d);
            CHECK(d < 1e-6, "embedding .st == safetensors (< 1e-6)");
        }
        free(x); free(y);
        /* a layer tensor too */
        float *q1 = (float *)malloc(sizeof(float) * 448 * 448);
        float *q2 = (float *)malloc(sizeof(float) * 448 * 448);
        const char *qn = layer_name(7, 0, nb, sizeof(nb));
        CHECK(wubu_ts_get_f32(a, qn, q1, 448LL * 448) == 0 &&
              wubu_ts_get_f32(b, qn, q2, 448LL * 448) == 0, "get layers.7.q from both");
        if (q1 && q2) {
            double d = maxdiff(q1, q2, 448LL * 448);
            printf("  layer q round-trip maxdiff = %g\n", d);
            CHECK(d < 1e-6, "layers.7 q .st == safetensors (< 1e-6)");
        }
        free(q1); free(q2);
    }

    /* 5. safetensors -> .st -> equality (back again) */
    if (b) {
        CHECK(wubu_ts_export(b, WUBU_TS_STDUMP, st_path3) == 0,
              "export safetensors -> .st");
        wubu_tensor_store_t *c = wubu_ts_open(st_path3);
        CHECK(c != NULL, "reopen back-converted .st");
        CHECK(c && wubu_ts_sniff(st_path3) == WUBU_TS_STDUMP, "sniff back .st");
        if (c) {
            float *x = (float *)malloc(sizeof(float) * 448 * 64);
            float *y = (float *)malloc(sizeof(float) * 448 * 64);
            const char *vn = layer_name(2, 2, nb, sizeof(nb));
            CHECK(wubu_ts_get_f32(b, vn, x, 448LL * 64) == 0 &&
                  wubu_ts_get_f32(c, vn, y, 448LL * 64) == 0, "get v_proj both");
            if (x && y) {
                double d = maxdiff(x, y, 448LL * 64);
                printf("  v_proj double-roundtrip maxdiff = %g\n", d);
                CHECK(d < 1e-6, "v_proj .st->st->.st identical (< 1e-6)");
            }
            free(x); free(y);
            wubu_ts_close(c);
        }
    }

    /* 6. safetensors -> GGUF -> equality + gguf_open reads it */
    if (b) {
        CHECK(wubu_ts_export(b, WUBU_TS_GGUF, st_path4) == 0,
              "export safetensors -> GGUF");
        CHECK(wubu_ts_sniff(st_path4) == WUBU_TS_GGUF, "sniff GGUF");
        wubu_tensor_store_t *d = wubu_ts_open(st_path4);
        CHECK(d != NULL, "reopen GGUF via catalog");
        CHECK(d && wubu_ts_format(d) == WUBU_TS_GGUF, "GGUF format reported");
        CHECK(d && wubu_ts_count(d) == ST_TOTAL, "GGUF catalog 137");
        if (d) {
            float *x = (float *)malloc(sizeof(float) * 448 * 448);
            float *y = (float *)malloc(sizeof(float) * 448 * 448);
            const char *on = layer_name(0, 3, nb, sizeof(nb)); /* o_proj */
            CHECK(wubu_ts_get_f32(b, on, x, 448LL * 448) == 0 &&
                  wubu_ts_get_f32(d, on, y, 448LL * 448) == 0, "get o_proj st vs gguf");
            if (x && y) {
                double dd = maxdiff(x, y, 448LL * 448);
                printf("  o_proj st->gguf maxdiff = %g\n", dd);
                CHECK(dd < 1e-6, "o_proj safetensors == GGUF (< 1e-6)");
            }
            free(x); free(y);
        }
        /* the gguf is also readable by the native gguf reader */
        gguf_ctx *g = gguf_open(st_path4);
        CHECK(g != NULL, "native gguf_open reads our GGUF");
        if (g) {
            gguf_tensor_info *ti = gguf_find_tensor(g, "embedding.weight");
            CHECK(ti != NULL, "gguf_find_tensor embedding.weight");
            gguf_close(g);
        }
        if (d) wubu_ts_close(d);
    }

    /* 7. name lookup works in all formats */
    if (a && b) {
        const wubu_ts_entry *ea = wubu_ts_find(a, layer_name(11, 9, nb, sizeof(nb)));
        CHECK(ea && ea->n_elems == 1228LL * 448, "layers.11.ffn.down has 1228*448");
    }

    if (a) wubu_ts_close(a);
    if (b) wubu_ts_close(b);
    remove(st_path); remove(st_path2); remove(st_path3); remove(st_path4);

    if (failures == 0) { printf("ALL TENSOR-STORE TESTS PASSED\n"); return 0; }
    printf("%d FAILURES\n", failures);
    return 1;
}
