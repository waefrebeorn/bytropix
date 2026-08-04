/*
 * test_dsa.c -- DSA indexer verification (coarse-to-fine block attention).
 *
 * Synthetic KV: 16 blocks x 8 keys, d=32, values d_out=8. Blocks point in
 * distinct random unit directions; block 3 carries magnitude 8.0 (so a query
 * equal to its mean dominates the indexer scores >10x), the rest magnitude
 * 0.05..0.15. Deterministic seed 48.
 */
#include "wubu_dsa.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define D 32
#define BS 8
#define NB 16
#define TOPK 3
#define D_OUT 8
#define SEED 48u

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

/* deterministic LCG */
static uint32_t rng_state;
static void rng_seed(uint32_t s) { rng_state = s; }
static uint32_t rng_next(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state; }
static float rng_f(void) { return ((float)(rng_next() >> 8) / 8388608.0f) - 1.0f; } /* [-1,1) */

static float dot_f(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* full softmax attention over the first nkeys keys of a flat key array */
static void full_attn(const float *q, const float *keys, int nkeys,
                      const float *vals, float *out) {
    float inv = 1.0f / sqrtf((float)D);
    float mx = -1e30f;
    for (int j = 0; j < nkeys; j++) {
        float s = dot_f(q, keys + (size_t)j * D, D) * inv;
        if (s > mx) mx = s;
    }
    float acc[D_OUT];
    for (int i = 0; i < D_OUT; i++) acc[i] = 0.0f;
    float den = 0.0f;
    for (int j = 0; j < nkeys; j++) {
        float w = expf(dot_f(q, keys + (size_t)j * D, D) * inv - mx);
        den += w;
        for (int i = 0; i < D_OUT; i++) acc[i] += w * vals[(size_t)j * D_OUT + i];
    }
    for (int i = 0; i < D_OUT; i++) out[i] = den > 0.0f ? acc[i] / den : 0.0f;
}

int main(void) {
    printf("=== test_dsa (DSA indexer) ===\n");

    /* --- synthetic KV: distinct block directions, block 3 dominant --- */
    static float keys[NB * BS * D];
    static float vals[NB * BS * D_OUT];
    static float mean[NB * D];
    static float *kptr[NB], *vptr[NB], *mptr[NB];

    rng_seed(SEED);
    for (int b = 0; b < NB; b++) {
        float dir[D], n2 = 0.0f;
        for (int i = 0; i < D; i++) { dir[i] = rng_f(); n2 += dir[i] * dir[i]; }
        n2 = sqrtf(n2);
        for (int i = 0; i < D; i++) dir[i] /= n2; /* unit direction */
        float mag = (b == 3) ? 8.0f
                             : 0.05f + 0.10f * ((float)(rng_next() % 1000) / 1000.0f);
        for (int j = 0; j < BS; j++) {
            for (int i = 0; i < D; i++)
                keys[(b * BS + j) * D + i] = dir[i] * mag + 0.01f * rng_f();
            for (int i = 0; i < D_OUT; i++)
                vals[(b * BS + j) * D_OUT + i] = rng_f(); /* [-1,1) */
        }
        for (int i = 0; i < D; i++) {
            float acc = 0.0f;
            for (int j = 0; j < BS; j++) acc += keys[(b * BS + j) * D + i];
            mean[b * D + i] = acc / (float)BS;
        }
        kptr[b] = &keys[b * BS * D];
        vptr[b] = &vals[b * BS * D_OUT];
        mptr[b] = &mean[b * D];
    }

    wubu_dsa_t *dsa = wubu_dsa_create(NB, BS, TOPK, D);
    CHECK(dsa != NULL, "create succeeds");

    /* --- (1) top-k correctness: exact k highest dot(query, mean) blocks,
       descending score order --- */
    {
        float q[D];
        for (int i = 0; i < D; i++) q[i] = rng_f();
        int sel[TOPK];
        CHECK(wubu_dsa_index(dsa, q, (const float *const *)mptr, sel) == TOPK,
              "index returns top_k");
        float sc[NB];
        int order[NB];
        for (int b = 0; b < NB; b++) { sc[b] = dot_f(q, mptr[b], D); order[b] = b; }
        /* brute-force sort: score desc, lower index first on ties */
        for (int i = 0; i < NB; i++)
            for (int j = i + 1; j < NB; j++)
                if (sc[order[j]] > sc[order[i]] ||
                    (sc[order[j]] == sc[order[i]] && order[j] < order[i])) {
                    int t = order[i]; order[i] = order[j]; order[j] = t;
                }
        int ok = 1;
        for (int c = 0; c < TOPK; c++) ok = ok && (sel[c] == order[c]);
        CHECK(ok, "top-k set and order match brute force");
        CHECK(sc[sel[0]] >= sc[sel[1]] && sc[sel[1]] >= sc[sel[2]],
              "selected blocks in descending score order");
        for (int c = 0; c < TOPK; c++)
            CHECK(sel[c] >= 0 && sel[c] < NB, "selected index in range");
    }

    /* --- (2) locality: query == block 3 mean selects block 3 first --- */
    {
        float q[D];
        for (int i = 0; i < D; i++) q[i] = mean[3 * D + i];
        int sel[TOPK];
        CHECK(wubu_dsa_index(dsa, q, (const float *const *)mptr, sel) == TOPK,
              "locality index returns top_k");
        CHECK(sel[0] == 3, "query equal to block 3 mean selects block 3 first");
    }

    /* --- (3) determinism: identical results across repeated calls --- */
    {
        float q[D];
        for (int i = 0; i < D; i++) q[i] = mean[3 * D + i];
        int s1[TOPK], s2[TOPK];
        float o1[D_OUT], o2[D_OUT];
        CHECK(wubu_dsa_index(dsa, q, (const float *const *)mptr, s1) == TOPK &&
              wubu_dsa_index(dsa, q, (const float *const *)mptr, s2) == TOPK,
              "determinism: index calls succeed");
        int same = 1;
        for (int c = 0; c < TOPK; c++) same = same && (s1[c] == s2[c]);
        CHECK(same, "index deterministic across calls");
        CHECK(wubu_dsa_attend(dsa, q, (const float *const *)kptr,
                              (const float *const *)vptr, o1, D_OUT) == 0 &&
              wubu_dsa_attend(dsa, q, (const float *const *)kptr,
                              (const float *const *)vptr, o2, D_OUT) == 0,
              "determinism: attend calls succeed");
        same = 1;
        for (int i = 0; i < D_OUT; i++) same = same && (o1[i] == o2[i]);
        CHECK(same, "attend deterministic across calls");
    }

    /* --- (4) approximation fidelity: dominant block => DSA attend within
       1e-2 L2 of full attention over ALL blocks --- */
    {
        float q[D];
        for (int i = 0; i < D; i++) q[i] = mean[3 * D + i];
        float s3 = dot_f(q, mptr[3], D), mx_other = -1e30f;
        for (int b = 0; b < NB; b++) {
            if (b == 3) continue;
            float s = dot_f(q, mptr[b], D);
            if (s > mx_other) mx_other = s;
        }
        CHECK(s3 > 10.0f * mx_other, "block 3 indexer score dominates (>10x)");
        float out_dsa[D_OUT], out_full[D_OUT];
        CHECK(wubu_dsa_attend(dsa, q, (const float *const *)kptr,
                              (const float *const *)vptr, out_dsa, D_OUT) == 0,
              "fidelity: attend returns 0");
        full_attn(q, keys, NB * BS, vals, out_full);
        float l2 = 0.0f;
        for (int i = 0; i < D_OUT; i++) {
            float dd = out_dsa[i] - out_full[i];
            l2 += dd * dd;
        }
        l2 = sqrtf(l2);
        CHECK(l2 <= 1e-2f, "DSA attend within 1e-2 L2 of full attention");
    }

    /* --- (5) output finite: random queries never produce NaN --- */
    {
        for (int t = 0; t < 16; t++) {
            float q[D];
            for (int i = 0; i < D; i++) q[i] = rng_f();
            int sel[TOPK];
            CHECK(wubu_dsa_index(dsa, q, (const float *const *)mptr, sel) == TOPK,
                  "finite: index succeeds");
            float out[D_OUT];
            CHECK(wubu_dsa_attend(dsa, q, (const float *const *)kptr,
                                  (const float *const *)vptr, out, D_OUT) == 0,
                  "finite: attend succeeds");
            int finite = 1;
            for (int i = 0; i < D_OUT; i++) finite = finite && isfinite(out[i]);
            CHECK(finite, "finite: attend output has no NaN/Inf");
        }
    }

    /* --- (6) API bounds: top_k > n_blocks clamps to n_blocks --- */
    {
        wubu_dsa_t *clamp = wubu_dsa_create(NB, BS, 20, D);
        CHECK(clamp != NULL, "clamp: create with top_k=20 succeeds");
        float q[D];
        for (int i = 0; i < D; i++) q[i] = rng_f();
        int sel[NB];
        int got = wubu_dsa_index(clamp, q, (const float *const *)mptr, sel);
        CHECK(got == NB, "clamp: index returns n_blocks (16), not 20");
        int seen[NB];
        for (int b = 0; b < NB; b++) seen[b] = 0;
        int ok = 1;
        for (int c = 0; c < got; c++) {
            ok = ok && sel[c] >= 0 && sel[c] < NB;
            if (ok) seen[sel[c]]++;
        }
        for (int b = 0; b < NB; b++) ok = ok && (seen[b] == 1);
        CHECK(ok, "clamp: every block selected exactly once");
        float sc[NB];
        for (int b = 0; b < NB; b++) sc[b] = dot_f(q, mptr[b], D);
        ok = 1;
        for (int c = 1; c < got; c++) ok = ok && (sc[sel[c - 1]] >= sc[sel[c]]);
        CHECK(ok, "clamp: full selection still score-descending");
        wubu_dsa_free(clamp);
    }

    /* --- API robustness --- */
    CHECK(wubu_dsa_create(0, BS, TOPK, D) == NULL, "create rejects n_blocks<=0");
    CHECK(wubu_dsa_create(NB, BS, 0, D) == NULL, "create rejects top_k<=0");
    CHECK(wubu_dsa_index(NULL, NULL, NULL, NULL) == -1, "index rejects null dsa");
    {
        float q[D], out[D_OUT];
        int sel[TOPK];
        for (int i = 0; i < D; i++) q[i] = 0.0f;
        CHECK(wubu_dsa_index(dsa, NULL, (const float *const *)mptr, sel) == -1,
              "index rejects null query");
        CHECK(wubu_dsa_attend(dsa, q, (const float *const *)kptr,
                              (const float *const *)vptr, out, 0) == -1,
              "attend rejects d_out<=0");
    }
    wubu_dsa_free(NULL); /* must be safe */

    wubu_dsa_free(dsa);

    if (failures == 0) { printf("ALL PASSED\n"); return 0; }
    printf("%d TEST(S) FAILED\n", failures);
    return 1;
}
