#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu.h"
#include "wubu_train.h"
#include "wubu_grow.h"
#include "wubu_backprop.h"

int main(void) {
    wubu_model_t m;
    wubu_buf_t b;
    wubu_train_t tr;
    memset(&m, 0, sizeof m);
    memset(&b, 0, sizeof b);
    memset(&tr, 0, sizeof tr);

    /* alloc model blocks like wubu_train_cli does */
    float *embedding = (float *)malloc(sizeof(float) * 16384 * 448);
    float *final_norm = (float *)malloc(sizeof(float) * 448);
    float **sel = (float **)calloc(BARUN_SELECTORS, sizeof(float *));
    wubu_block_t *blocks = (wubu_block_t *)calloc(BARUN_LAYERS, sizeof(wubu_block_t));
    for (int i = 0; i < BARUN_SELECTORS; i++) sel[i] = (float *)malloc(sizeof(float) * 448);
    for (int i = 0; i < BARUN_LAYERS; i++) {
        blocks[i].q_proj   = (float *)calloc(448*448, sizeof(float));
        blocks[i].k_proj   = (float *)calloc(448*64, sizeof(float));
        blocks[i].v_proj   = (float *)calloc(448*64, sizeof(float));
        blocks[i].o_proj   = (float *)calloc(448*448, sizeof(float));
        blocks[i].g_proj   = (float *)calloc(448*448, sizeof(float));
        blocks[i].q_norm   = (float *)calloc(64, sizeof(float));
        blocks[i].k_norm   = (float *)calloc(64, sizeof(float));
        blocks[i].attn_norm= (float *)calloc(448, sizeof(float));
        blocks[i].gate_up  = (float *)calloc(448*2456, sizeof(float));
        blocks[i].down     = (float *)calloc(1228*448, sizeof(float));
        blocks[i].ffn_norm = (float *)calloc(448, sizeof(float));
    }
    wubu_model_init(&m, embedding, final_norm, blocks, sel);
    m.n_layers = 2;

    if (wubu_buf_alloc(&b, BARUN_MAX_SEQ) != 0) { printf("buf OOM\n"); return 1; }
    if (wubu_train_init(&tr, &m) != 0) { printf("train OOM\n"); return 1; }

    printf("pre-grow: n_layers=%d\n", m.n_layers);

    /* print block q_proj addresses before grow */
    for (int i = 0; i < 12; i++) {
        printf("  block[%d].q_proj = %p\n", i, (void*)m.blocks[i].q_proj);
    }

    int pos = 1;
    for (int g = 0; g < 4; g++) {
        int pre = m.n_layers;
        printf("--- GROW %d: pre=%d, pos=%d ---\n", g+1, pre, pos);
        int ok1 = wubu_grow_insert_block(&m, pos);
        int ok2 = wubu_train_grow(&tr, pos, pre);
        printf("  grow_insert=%d train_grow=%d -> n_layers=%d\n", ok1, ok2, m.n_layers);
        for (int i = 0; i < 12; i++) {
            printf("  block[%d].q_proj = %p\n", i, (void*)m.blocks[i].q_proj);
        }
    }

    printf("post-grow: n_layers=%d, freeing...\n", m.n_layers);
    wubu_train_free(&tr);
    printf("  train_free done\n");
    wubu_free(&m, &b);
    printf("  wubu_free done\n");
    /* wubu_free already freed embedding, final_norm, and selectors */
    /* free(sel);  -- already freed by wubu_free */
    /* free(blocks); -- blocks are freed by wubu_free */
    printf("ALL CLEAN\n");
    return 0;
}
