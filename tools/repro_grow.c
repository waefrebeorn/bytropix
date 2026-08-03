/* repro_grow.c -- the exact CLI growth sequence repro (the progressive
 * deepening: pos_g = n_layers / 2, seven middle insertions 2->9).
 * Standalone: no training math, just the growth operator + the train
 * state -- built for ASan so the double-free names itself instantly. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_barun.h"
#include "wubu_grow.h"
#include "wubu_barun_train.h"
#include "wubu_barun_backprop.h"

static float *mk(int n, unsigned *seed)
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
    blk.q_proj    = mk(BARUN_DIM * BARUN_HEADS * 64, seed);
    blk.k_proj    = mk(BARUN_DIM * BARUN_KV_HEADS * 64, seed);
    blk.v_proj    = mk(BARUN_DIM * BARUN_KV_HEADS * 64, seed);
    blk.o_proj    = mk(BARUN_DIM * BARUN_HEADS * 64, seed);
    blk.g_proj    = mk(BARUN_DIM * BARUN_HEADS * 64, seed);
    blk.q_norm    = mk(BARUN_KV_HEADS * 64, seed);
    blk.k_norm    = mk(BARUN_KV_HEADS * 64, seed);
    blk.attn_norm = mk(BARUN_DIM, seed);
    blk.gate_up   = mk(BARUN_DIM * BARUN_FFN_DIM * 2, seed);
    blk.down      = mk(BARUN_FFN_DIM * BARUN_DIM, seed);
    blk.ffn_norm  = mk(BARUN_DIM, seed);
    return blk;
}

int main(void)
{
    unsigned seed = 99;
    barun_block_t blocks[BARUN_LAYERS];
    for (int i = 0; i < BARUN_LAYERS; i++) blocks[i] = make_block(&seed);
    float *embedding = mk(BARUN_VOCAB * BARUN_DIM, &seed);
    float *final_norm = mk(BARUN_DIM, &seed);
    float *sel[BARUN_SELECTORS];
    for (int i = 0; i < BARUN_SELECTORS; i++) sel[i] = mk(BARUN_DIM, &seed);

    barun_model_t m;
    if (barun_model_init(&m, embedding, final_norm, blocks, sel) != 0) {
        printf("  model init FAIL\n"); return 1;
    }
    barun_train_t tr;
    if (barun_train_init(&tr, &m) != 0) { printf("  train init FAIL\n"); return 1; }
    barun_bp_t bp;
    if (barun_bp_alloc(&bp, 16) != 0) { printf("  bp alloc FAIL\n"); return 1; }
    barun_buf_t b;
    if (barun_buf_alloc(&b, 16) != 0) { printf("  buf alloc FAIL\n"); return 1; }
    barun_train_cfg_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.lr = 1e-3f; cfg.muon_lr = 1e-3f; cfg.adam_lr = 1e-3f;
    cfg.weight_decay = 0.1f; cfg.grad_clip = 0.0f;

    uint16_t toks[8];
    for (int i = 0; i < 8; i++) toks[i] = (uint16_t)(i * 3 % 512 + 10);

    m.n_layers = 2;
    for (int n = 2; n <= 8; n++) {           /* grow#1..#7 (2->9) */
        int pos_g = n / 2;                    /* the CLI's middle-insert */
        fprintf(stderr, "grow#%d: n=%d pos=%d\n", n - 1, n, pos_g);
        if (!wubu_grow_insert_block(&m, pos_g)) { printf("  insert FAIL\n"); return 1; }
        if (!wubu_train_grow(&tr, pos_g, n + 1)) { printf("  train_grow FAIL\n"); return 1; }
        /* FIFTY tiny training steps on the grown model (the real CLI
         * runs grow_check=50 steps between grows; the crash needs the
         * accumulation -- one step per grow was clean) */
        for (int s = 0; s < 50; s++) {
            barun_train_zero_grad(&tr);
            float loss = barun_bp_forward(&m, &b, &bp, toks, 8);
            barun_bp_backward(&m, &b, &bp, &tr, toks, 8);
            barun_bp_muon_step(&m, &tr, &cfg, (uint32_t)(n * 100 + s));
            if (s % 25 == 0)
                fprintf(stderr, "  step %d/%d: loss=%.4f\n", s, 50, loss);
        }
        fprintf(stderr, "  ok: n_layers=%d\n", m.n_layers);
    }
    fprintf(stderr, "ALL 7 GROWTHS + 350 STEPS OK\n");
    return 0;
}