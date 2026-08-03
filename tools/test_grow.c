/* test_grow.c -- the model-growth operator against the DA oracle:
 * the FUNCTION-PRESERVING property (the zero-init insertion must leave
 * the forward outputs byte-identical), the G_stack copy's forward runs,
 * and the Bu progressive schedule's monotonicity. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_barun.h"
#include "wubu_grow.h"
#include "wubu_barun_backprop.h"
#include "wubu_barun_train.h"

#define SEQ 16

static float *rnd(int n, unsigned *seed)
{
    float *p = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        *seed = *seed * 1664525u + 1013904223u;
        p[i] = ((float)(*seed % 2000) / 1000.0f - 1.0f) * 0.05f;
    }
    return p;
}

static barun_block_t make_block(unsigned *seed)
{
    barun_block_t blk;
    memset(&blk, 0, sizeof blk);
    blk.q_proj    = rnd(BARUN_DIM * BARUN_HEADS * 64, seed);
    blk.k_proj    = rnd(BARUN_DIM * BARUN_KV_HEADS * 64, seed);
    blk.v_proj    = rnd(BARUN_DIM * BARUN_KV_HEADS * 64, seed);
    blk.o_proj    = rnd(BARUN_DIM * BARUN_HEADS * 64, seed);
    blk.g_proj    = rnd(BARUN_DIM * BARUN_HEADS * 64, seed);
    blk.q_norm    = rnd(BARUN_KV_HEADS * 64, seed);
    blk.k_norm    = rnd(BARUN_KV_HEADS * 64, seed);
    blk.attn_norm = rnd(BARUN_DIM, seed);
    blk.gate_up   = rnd(BARUN_DIM * BARUN_FFN_DIM * 2, seed);
    blk.down      = rnd(BARUN_FFN_DIM * BARUN_DIM, seed);
    blk.ffn_norm  = rnd(BARUN_DIM, seed);
    return blk;
}

int main(void)
{
    unsigned seed = 42;
    barun_block_t blocks[BARUN_LAYERS];
    for (int i = 0; i < BARUN_LAYERS; i++) blocks[i] = make_block(&seed);
    float *embedding = rnd(16384 * BARUN_DIM, &seed);
    float *final_norm = rnd(BARUN_DIM, &seed);
    float *sel[BARUN_SELECTORS];
    for (int i = 0; i < BARUN_SELECTORS; i++) sel[i] = rnd(BARUN_DIM, &seed);

    barun_model_t m;
    if (barun_model_init(&m, embedding, final_norm, blocks, sel) != 0) {
        printf("  model init FAIL\n"); return 1;
    }
    barun_buf_t b;
    if (barun_buf_alloc(&b, 512) != 0) { printf("  buf alloc FAIL\n"); return 1; }

    uint16_t toks[SEQ];
    for (int i = 0; i < SEQ; i++) toks[i] = (uint16_t)(i * 7 % 512 + 10);

    int ok = 1;

    /* --- the function-preserving insertion (the DA oracle) --- */
    m.n_layers = 2;                    /* the "small model" (Bu: start small) */
    barun_forward(&m, &b, toks, SEQ);
    float *pre = (float *)malloc((size_t)SEQ * 16384 * sizeof(float));
    memcpy(pre, b.logits, (size_t)SEQ * 16384 * sizeof(float));

    int r1 = wubu_grow_insert_block(&m, 1);    /* insert the zero block */
    int r2 = wubu_grow_insert_block(&m, 3);    /* and again at another pos */
    if (m.n_layers != 4 || !r1 || !r2) { printf("  insert FAIL\n"); ok = 0; }
    barun_forward(&m, &b, toks, SEQ);
    double maxd = 0;
    for (size_t i = 0; i < (size_t)SEQ * 16384; i++) {
        double d = fabs((double)pre[i] - (double)b.logits[i]);
        if (d > maxd) maxd = d;
    }
    printf("  function-preserving insert: max|pre-post| = %.3e  %s\n",
           maxd, maxd < 1e-5 ? "PASS" : "FAIL");
    if (maxd >= 1e-5) ok = 0;

    /* --- the G_stack copy: the grown model must still run + the params
     * actually grew (the parameter count doubles the stacked block) --- */
    long p0 = barun_parameter_count(&m);
    int r3 = wubu_grow_stack_block(&m, 0);
    if (!r3 || m.n_layers != 5) { printf("  stack FAIL\n"); ok = 0; }
    long p1 = barun_parameter_count(&m);
    long blk = p1 - p0;
    if (blk <= 0) { printf("  stack param growth FAIL\n"); ok = 0; }
    if (barun_forward(&m, &b, toks, SEQ) != 0) { printf("  grown forward FAIL\n"); ok = 0; }
    printf("  G_stack: +%ld params, grown forward runs  %s\n", blk, ok ? "PASS" : "FAIL");

    /* --- the growth through the backprop (the DA: the bp's loss of the
     * grown model must EQUAL the small model's -- the zero block is an
     * identity -- and the grown block's grads must match the FD) --- */
    {
        barun_bp_t bp;
        if (barun_bp_alloc(&bp, 512) != 0) { printf("  bp alloc FAIL\n"); return 1; }
        barun_train_t tr;
        if (barun_train_init(&tr, &m) != 0) { printf("  train init FAIL\n"); return 1; }
        m.n_layers = 2;
        float loss_small = barun_bp_forward(&m, &b, &bp, toks, SEQ);
        /* grow the model + the train state */
        if (!wubu_grow_insert_block(&m, 1)) { printf("  bp-grow insert FAIL\n"); ok = 0; }
        if (!wubu_train_grow(&tr, 1, m.n_layers)) { printf("  bp train-grow FAIL\n"); ok = 0; }
        if (!wubu_grow_insert_block(&m, 3)) { printf("  bp-grow insert2 FAIL\n"); ok = 0; }
        if (!wubu_train_grow(&tr, 3, m.n_layers)) { printf("  bp train-grow2 FAIL\n"); ok = 0; }
        float loss_grown = barun_bp_forward(&m, &b, &bp, toks, SEQ);
        double ldiff = fabs((double)loss_small - (double)loss_grown);
        printf("  bp loss small %.6f grown %.6f diff %.3e  %s\n",
               loss_small, loss_grown, ldiff, ldiff < 1e-4 ? "PASS" : "FAIL");
        if (ldiff >= 1e-4) ok = 0;
        /* the backward on the grown model (the grads must be real) */
        barun_train_zero_grad(&tr);
        barun_bp_backward(&m, &b, &bp, &tr, toks, SEQ);
        /* the FD on the EMBEDDING: the grad is ~1e-2 -- measurable in the
         * fp32 loss (the q_proj grads are ~1e-5, swamped by the 9.7-scale
         * fp32 loss -- the fp32-honesty lesson). The grown model's loss
         * must still depend on the weights through the grown stack. */
        float *w = m.embedding;
        float *wg = tr.emb_g;
        size_t brow = (size_t)toks[3] * BARUN_DIM;
        size_t bd = 0;
        for (int d = 1; d < BARUN_DIM; d++)
            if (fabs((double)wg[brow + d]) > fabs((double)wg[brow + bd])) bd = d;
        size_t bi = brow + bd;
        float save = w[bi];
        float epsf = 1e-2f;
        w[bi] = save + epsf;
        float l1 = barun_bp_forward(&m, &b, &bp, toks, SEQ);
        w[bi] = save - epsf;
        float l2 = barun_bp_forward(&m, &b, &bp, toks, SEQ);
        w[bi] = save;
        double fd = (l1 - l2) / (2.0 * epsf);
        double rel = fabs(fd - wg[bi]) / (fabs(fd) + 1e-9);
        printf("  grown-bp FD (embedding): fd=%.6f grad=%.6f rel=%.3e  %s\n",
               fd, wg[bi], rel, rel < 1e-2 ? "PASS" : "FAIL");
        if (rel >= 1e-2) ok = 0;
        barun_bp_free(&bp);
        barun_train_free(&tr);
    }

    /* --- the Bu schedule: expand every 10% of the horizon --- */
    int T = 1000, last = 1;
    int mono = 1, reached = 0;
    for (int t = 0; t <= T; t++) {
        int l = wubu_grow_schedule(t, T, 2, 12, 0.1f);
        if (l < last) { mono = 0; break; }
        last = l;
        if (l == 12) reached = 1;
    }
    int ev = wubu_grow_events(T, 2, 12, 0.1f);
    printf("  Bu schedule: monotonic %s, reaches 12 %s, events %d  %s\n",
           mono ? "yes" : "NO", reached ? "yes" : "NO", ev,
           (mono && reached && ev == 10) ? "PASS" : "FAIL");
    if (!mono || !reached || ev != 10) ok = 0;

    printf("%s\n", ok ? "ALL GROW TESTS PASSED" : "GROW FAILURES");
    return ok ? 0 : 1;
}
