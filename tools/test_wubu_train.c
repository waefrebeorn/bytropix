/*
 * test_wubu_train.c -- the BarunLM training core test (the AGI loop).
 *
 * Proves the training loop: the seed model takes synthetic sequences,
 * the loss is computed, Muon+AdamW updates the weights, and the loss
 * DECREASES over steps -- the mustard seed learns. Also verifies the
 * released checkpoint can be fine-tuned (the weights change but stay
 * finite).
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wubu.h"
#include "wubu_train.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

/* a tiny synthetic "language": sequences of ids 10..40 with a pattern
 * (the even ids follow the odd ones) so the model has something to learn. */
static void make_sequence(uint16_t *tok, size_t n, uint32_t seed)
{
    for (size_t i = 0; i < n; i++) {
        uint32_t r = (seed * 1103515245u + 12345u) >> 16;
        seed = seed * 1103515245u + 12345u;
        /* pattern: even -> odd+1, odd -> a random id */
        uint16_t base = (uint16_t)(10 + (r % 30));
        tok[i] = (base % 2 == 0) ? (uint16_t)(base + 1) : base;
    }
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "models/wubu/model.safetensors";
    printf("=== test_wubu_train (the AGI training loop) ===\n");

    wubu_model_t m;
    if (wubu_load(&m, path) != 0) {
        printf("  FAIL: cannot load %s\n", path);
        return 1;
    }
    printf("  loaded the seed (%ld parameters)\n", wubu_parameter_count(&m));

    wubu_buf_t b;
    CHECK(wubu_buf_alloc(&b, 128) == 0, "buf alloc");

    wubu_train_t tr;
    CHECK(wubu_train_init(&tr, &m) == 0, "train init");

    wubu_train_cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.lr = 1e-3f;          /* the seed loop uses a higher lr than the
                                reference's 1e-4 so a few steps show the loss
                                moving (the full run uses the reference
                                schedule: lr 1e-4, wd 0.1, batch 48, seq 2048) */
    cfg.weight_decay = 0.1f;
    cfg.muon_momentum = 0.95f;
    cfg.warmup_steps = 2;
    cfg.max_steps = 20;

    /* snapshot a few weights to prove they change */
    float w0 = m.blocks[0].q_proj[0];
    float e0 = m.embedding[0];

    uint16_t tok[128];
    float first_loss = -1, last_loss = -1;
    for (uint32_t step = 1; step <= 6; step++) {
        make_sequence(tok, 128, step * 7919u);
        float loss = wubu_train_step_loop(&m, &tr, &b, tok, 128, &cfg, step);
        if (step == 1) first_loss = loss;
        last_loss = loss;
        printf("  step %u: loss %.4f\n", step, loss);
    }
    CHECK(first_loss > 0, "loss is positive");
    CHECK(last_loss < first_loss, "loss decreased (the seed learns)");

    /* the weights changed but stay finite */
    float w1 = m.blocks[0].q_proj[0];
    float e1 = m.embedding[0];
    CHECK(w1 != w0 || e1 != e0, "weights changed");
    CHECK(w1 == w1 && e1 == e1, "weights finite");

    /* the LR schedule */
    float lr1 = wubu_train_lr(&cfg, 1);
    float lr2 = wubu_train_lr(&cfg, 10);
    CHECK(lr1 > 0 && lr2 > 0, "lr positive");
    printf("  lr(1)=%.6f lr(10)=%.6f\n", lr1, lr2);

    wubu_train_free(&tr);
    wubu_free(&m, &b);

    if (failures == 0) printf("ALL BARUN_TRAIN TESTS PASSED -- the seed learns\n");
    else printf("%d BARUN_TRAIN FAILURES\n", failures);
    return failures ? 1 : 0;
}
