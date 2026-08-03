/* test_width.c -- the width expansion structural DA oracle:
 * every expanded weight keeps the old block top-left EXACTLY, zeroes
 * the new rows/columns, the embedding's right half is zero, and the
 * norms' new half sits at the identity scale. (The engine-run DA
 * -- max|pre-post| = 0 -- needs the dynamic-dims refactor; this test
 * pins the weight-level invariants that refactor consumes.) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_width.h"

static float *mk(int n, unsigned *seed)
{
    float *p = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        *seed = *seed * 1664525u + 1013904223u;
        p[i] = ((float)(*seed % 2000) / 1000.0f - 1.0f) * 0.05f;
    }
    return p;
}

int main(void)
{
    unsigned seed = 7;
    int ok = 1;
    int fails = 0;

    /* build a small model (2 active layers) with known weights */
    barun_block_t blocks[BARUN_LAYERS];
    memset(blocks, 0, sizeof blocks);
    for (int i = 0; i < BARUN_LAYERS; i++) {
        blocks[i].q_proj    = mk(BARUN_DIM * BARUN_HEADS * 64, &seed);
        blocks[i].k_proj    = mk(BARUN_DIM * BARUN_KV_HEADS * 64, &seed);
        blocks[i].v_proj    = mk(BARUN_DIM * BARUN_KV_HEADS * 64, &seed);
        blocks[i].o_proj    = mk(BARUN_DIM * BARUN_HEADS * 64, &seed);
        blocks[i].g_proj    = mk(BARUN_DIM * BARUN_HEADS * 64, &seed);
        blocks[i].q_norm    = mk(BARUN_KV_HEADS * 64, &seed);
        blocks[i].k_norm    = mk(BARUN_KV_HEADS * 64, &seed);
        blocks[i].attn_norm = mk(BARUN_DIM, &seed);
        blocks[i].gate_up   = mk(BARUN_DIM * BARUN_FFN_DIM * 2, &seed);
        blocks[i].down      = mk(BARUN_FFN_DIM * BARUN_DIM, &seed);
        blocks[i].ffn_norm  = mk(BARUN_DIM, &seed);
    }
    float *embedding = mk(BARUN_VOCAB * BARUN_DIM, &seed);
    float *final_norm = mk(BARUN_DIM, &seed);
    float *sel[BARUN_SELECTORS];
    for (int i = 0; i < BARUN_SELECTORS; i++) sel[i] = mk(BARUN_DIM, &seed);

    barun_model_t m;
    if (barun_model_init(&m, embedding, final_norm, blocks, sel) != 0) {
        printf("  model init FAIL\n"); return 1;
    }
    m.n_layers = 2;

    /* snapshot the old weights for the exactness checks (the expansion
     * FREES the originals -- barun_model_init references, doesn't copy) */
    float *old_q = (float *)malloc((size_t)BARUN_DIM * BARUN_DIM * sizeof(float));
    float *old_k = (float *)malloc((size_t)BARUN_DIM * BARUN_KV_HEADS * BARUN_HEAD_DIM * sizeof(float));
    float *old_gu = (float *)malloc((size_t)BARUN_DIM * BARUN_FFN_DIM * 2 * sizeof(float));
    float *old_dn = (float *)malloc((size_t)BARUN_FFN_DIM * BARUN_DIM * sizeof(float));
    float *old_emb = (float *)malloc((size_t)16 * BARUN_DIM * sizeof(float));
    float *old_an = (float *)malloc((size_t)BARUN_DIM * sizeof(float));
    float *old_fn = (float *)malloc((size_t)BARUN_DIM * sizeof(float));
    memcpy(old_q, m.blocks[0].q_proj, (size_t)BARUN_DIM * BARUN_DIM * sizeof(float));
    memcpy(old_k, m.blocks[0].k_proj, (size_t)BARUN_DIM * BARUN_KV_HEADS * BARUN_HEAD_DIM * sizeof(float));
    memcpy(old_gu, m.blocks[0].gate_up, (size_t)BARUN_DIM * BARUN_FFN_DIM * 2 * sizeof(float));
    memcpy(old_dn, m.blocks[0].down, (size_t)BARUN_FFN_DIM * BARUN_DIM * sizeof(float));
    memcpy(old_emb, m.embedding, (size_t)16 * BARUN_DIM * sizeof(float));
    memcpy(old_an, m.blocks[0].attn_norm, (size_t)BARUN_DIM * sizeof(float));
    memcpy(old_fn, m.final_norm, (size_t)BARUN_DIM * sizeof(float));

    if (!wubu_width_expand(&m)) { printf("  expand FAIL\n"); return 1; }

    const int D = BARUN_DIM, D2 = BARUN_DIM * 2;
    const int F = BARUN_FFN_DIM, F2 = BARUN_FFN_DIM * 2;

    /* 1. q_proj: [D,D] -> [D2,D2], top-left exact, new rows/cols zero */
    for (int r = 0; r < D; r++)
        for (int c = 0; c < D; c++)
            if (m.blocks[0].q_proj[r * D2 + c] != old_q[r * D + c]) fails++;
    for (int r = 0; r < D2; r++)
        for (int c = 0; c < D2; c++)
            if ((r >= D || c >= D) && m.blocks[0].q_proj[r * D2 + c] != 0.0f) fails++;
    if (fails) { printf("  q_proj layout FAIL (%d)\n", fails); ok = 0; fails = 0; }

    /* 2. k_proj: [D, 64] -> [D2, 64], old rows exact, new rows zero */
    const int kv = BARUN_KV_HEADS * BARUN_HEAD_DIM;
    for (int r = 0; r < D; r++)
        for (int c = 0; c < kv; c++)
            if (m.blocks[0].k_proj[r * kv + c] != old_k[r * kv + c]) fails++;
    for (int r = D; r < D2; r++)
        for (int c = 0; c < kv; c++)
            if (m.blocks[0].k_proj[r * kv + c] != 0.0f) fails++;
    if (fails) { printf("  k_proj layout FAIL (%d)\n", fails); ok = 0; fails = 0; }

    /* 3. gate_up: [D, 2F] -> [D2, 2F2], top-left exact, rest zero */
    for (int r = 0; r < D; r++)
        for (int c = 0; c < 2 * F; c++)
            if (m.blocks[0].gate_up[r * 2 * F2 + c] != old_gu[r * 2 * F + c]) fails++;
    for (int r = 0; r < D2; r++)
        for (int c = 0; c < 2 * F2; c++)
            if ((r >= D || c >= 2 * F) && m.blocks[0].gate_up[r * 2 * F2 + c] != 0.0f) fails++;
    if (fails) { printf("  gate_up layout FAIL (%d)\n", fails); ok = 0; fails = 0; }

    /* 4. down: [F, D] -> [F2, D2], top-left exact, rest zero */
    for (int r = 0; r < F; r++)
        for (int c = 0; c < D; c++)
            if (m.blocks[0].down[r * D2 + c] != old_dn[r * D + c]) fails++;
    for (int r = 0; r < F2; r++)
        for (int c = 0; c < D2; c++)
            if ((r >= F || c >= D) && m.blocks[0].down[r * D2 + c] != 0.0f) fails++;
    if (fails) { printf("  down layout FAIL (%d)\n", fails); ok = 0; fails = 0; }

    /* 5. the embedding: left half exact, right half zero */
    for (int r = 0; r < 16; r++) {
        for (int c = 0; c < D; c++)
            if (m.embedding[r * D2 + c] != old_emb[r * D + c]) fails++;
        for (int c = D; c < D2; c++)
            if (m.embedding[r * D2 + c] != 0.0f) fails++;
    }
    if (fails) { printf("  embedding layout FAIL (%d)\n", fails); ok = 0; fails = 0; }

    /* 6. the norms: first half exact, second half at the identity scale */
    for (int i = 0; i < D; i++)
        if (m.blocks[0].attn_norm[i] != old_an[i]) fails++;
    for (int i = D; i < D2; i++)
        if (m.blocks[0].attn_norm[i] != 1.0f) fails++;
    if (fails) { printf("  attn_norm layout FAIL (%d)\n", fails); ok = 0; fails = 0; }
    for (int i = 0; i < D; i++)
        if (m.final_norm[i] != old_fn[i]) fails++;
    for (int i = D; i < D2; i++)
        if (m.final_norm[i] != 1.0f) fails++;
    if (fails) { printf("  final_norm layout FAIL (%d)\n", fails); ok = 0; fails = 0; }

    printf("  width expand: every weight doubled with the old block EXACT, "
           "new rows/cols zero  %s\n", ok ? "PASS" : "FAIL");

    printf("%s\n", ok ? "ALL WIDTH TESTS PASSED" : "WIDTH FAILURES");
    return ok ? 0 : 1;
}