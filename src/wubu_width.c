/* wubu_width.c -- the width expansion (Net2Net dynamic-dims refactor).
 *
 * The ZERO-PADDING identity: every expanded weight matrix keeps the
 * old block in its top-left corner EXACTLY (no scaling) and zeroes the
 * new rows and new columns. With the hidden stream x' = [x; 0], any
 * expanded matrix gives W'x' = [Wx; 0] -- the left half is the old
 * output exactly, the right half stays zero, so a stack of expanded
 * blocks computes the identical left-half stream.
 *
 * The attention's new heads get zero q/k/v, so they contribute zero
 * through the o_proj (whose new columns are zeroed). The norms' new
 * half sits at the identity scale (1.0); with zero activations the
 * normalized value is 0 regardless, so the stream stays zero. */
#include <stdlib.h>
#include <string.h>
#include "wubu_width.h"

/* Helpers: expand a [rows, cols] matrix to [2*rows, 2*cols]:
 * top-left = old EXACT, everything else zero. */
static int expand_mat(float **dst, const float *src, int rows, int cols)
{
    float *n = (float *)calloc((size_t)(2 * rows) * (2 * cols), sizeof(float));
    if (!n) return 0;
    for (int r = 0; r < rows; r++)
        memcpy(&n[r * 2 * cols], &src[r * cols], (size_t)cols * sizeof(float));
    *dst = n;
    return 1;
}

/* Expand a [rows, cols] matrix to [2*rows, cols] (the width dim is the
 * ROWS here -- e.g. k_proj [448, 64] -> [896, 64]): old rows exact,
 * new rows zero. */
static int expand_rows(float **dst, const float *src, int rows, int cols)
{
    float *n = (float *)calloc((size_t)(2 * rows) * cols, sizeof(float));
    if (!n) return 0;
    for (int r = 0; r < rows; r++)
        memcpy(&n[r * cols], &src[r * cols], (size_t)cols * sizeof(float));
    *dst = n;
    return 1;
}

/* Expand a [rows, cols] vector-norm to [2*rows]: old exact, new half
 * at the identity scale (1.0). */
static int expand_norm(float **dst, const float *src, int n)
{
    float *nd = (float *)calloc((size_t)(2 * n), sizeof(float));
    if (!nd) return 0;
    memcpy(nd, src, (size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) nd[n + i] = 1.0f;
    *dst = nd;
    return 1;
}

int wubu_width_expand(barun_model_t *m)
{
    if (!m || m->n_layers <= 0) return 0;

    /* the embedding: [vocab, 448] -> [vocab, 896], right half zero */
    {
        float *e = (float *)calloc((size_t)BARUN_VOCAB * BARUN_DIM * 2, sizeof(float));
        if (!e) return 0;
        for (int r = 0; r < BARUN_VOCAB; r++)
            memcpy(&e[r * BARUN_DIM * 2], &m->embedding[r * BARUN_DIM],
                   (size_t)BARUN_DIM * sizeof(float));
        free(m->embedding);
        m->embedding = e;
    }
    /* the final norm + the selectors: [448] -> [896] (identity half) */
    if (!expand_norm(&m->final_norm, m->final_norm, BARUN_DIM)) return 0;
    for (int s = 0; s < BARUN_SELECTORS; s++)
        if (!expand_norm(&m->selectors[s], m->selectors[s], BARUN_DIM)) return 0;

    for (int l = 0; l < m->n_layers; l++) {
        barun_block_t *b = &m->blocks[l];
        /* q/o/g: [448, 448] -> [896, 896] (square) */
        if (!expand_mat(&b->q_proj, b->q_proj, BARUN_DIM, BARUN_DIM)) return 0;
        if (!expand_mat(&b->o_proj, b->o_proj, BARUN_DIM, BARUN_DIM)) return 0;
        if (!expand_mat(&b->g_proj, b->g_proj, BARUN_DIM, BARUN_DIM)) return 0;
        /* k/v: [448, 64] -> [896, 64] (rows = the width) */
        if (!expand_rows(&b->k_proj, b->k_proj, BARUN_DIM,
                         BARUN_KV_HEADS * BARUN_HEAD_DIM)) return 0;
        if (!expand_rows(&b->v_proj, b->v_proj, BARUN_DIM,
                         BARUN_KV_HEADS * BARUN_HEAD_DIM)) return 0;
        /* gate_up: [448, 2*1228] -> [896, 2*2456] */
        if (!expand_mat(&b->gate_up, b->gate_up, BARUN_DIM, BARUN_FFN_DIM * 2)) return 0;
        /* down: [1228, 448] -> [2456, 896] (square-ish; rows = ffn) */
        if (!expand_mat(&b->down, b->down, BARUN_FFN_DIM, BARUN_DIM)) return 0;
        /* the norms: [448] -> [896] */
        if (!expand_norm(&b->attn_norm, b->attn_norm, BARUN_DIM)) return 0;
        if (!expand_norm(&b->ffn_norm, b->ffn_norm, BARUN_DIM)) return 0;
        if (!expand_norm(&b->q_norm, b->q_norm, BARUN_KV_HEADS * BARUN_HEAD_DIM)) return 0;
        if (!expand_norm(&b->k_norm, b->k_norm, BARUN_KV_HEADS * BARUN_HEAD_DIM)) return 0;
    }
    return 1;
}