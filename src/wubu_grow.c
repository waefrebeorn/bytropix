/* wubu_grow.c -- the model-growth operator. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_grow.h"

/* the per-block weight byte size (all the block buffers) */
static size_t block_bytes(void)
{
    size_t s = 0;
    s += (size_t)BARUN_DIM * BARUN_HEADS * 64 * sizeof(float);      /* q */
    s += (size_t)BARUN_DIM * BARUN_KV_HEADS * 64 * sizeof(float);   /* k */
    s += (size_t)BARUN_DIM * BARUN_KV_HEADS * 64 * sizeof(float);   /* v */
    s += (size_t)BARUN_DIM * BARUN_HEADS * 64 * sizeof(float);      /* o */
    s += (size_t)BARUN_DIM * BARUN_HEADS * 64 * sizeof(float);      /* g */
    s += (size_t)BARUN_KV_HEADS * 64 * sizeof(float);               /* q_norm */
    s += (size_t)BARUN_KV_HEADS * 64 * sizeof(float);               /* k_norm */
    s += (size_t)BARUN_DIM * sizeof(float);                         /* attn_norm */
    s += (size_t)BARUN_DIM * BARUN_FFN_DIM * 2 * sizeof(float);         /* gate_up */
    s += (size_t)BARUN_FFN_DIM * BARUN_DIM * sizeof(float);             /* down */
    s += (size_t)BARUN_DIM * sizeof(float);                         /* ffn_norm */
    return s;
}

static barun_block_t zero_block(void)
{
    barun_block_t z;
    memset(&z, 0, sizeof z);
    size_t b = block_bytes();
    float *mem = (float *)calloc(b, 1);
    if (!mem) { memset(&z, 0, sizeof z); return z; }
    /* the buffer layout: q k v o g qn kn an gu d fn */
    float *p = mem;
    z.q_proj   = p; p += (size_t)BARUN_DIM * BARUN_HEADS * 64;
    z.k_proj   = p; p += (size_t)BARUN_DIM * BARUN_KV_HEADS * 64;
    z.v_proj   = p; p += (size_t)BARUN_DIM * BARUN_KV_HEADS * 64;
    z.o_proj   = p; p += (size_t)BARUN_DIM * BARUN_HEADS * 64;
    z.g_proj   = p; p += (size_t)BARUN_DIM * BARUN_HEADS * 64;
    z.q_norm   = p; p += (size_t)BARUN_KV_HEADS * 64;
    z.k_norm   = p; p += (size_t)BARUN_KV_HEADS * 64;
    z.attn_norm= p; p += (size_t)BARUN_DIM;
    z.gate_up  = p; p += (size_t)BARUN_DIM * BARUN_FFN_DIM * 2;
    z.down     = p; p += (size_t)BARUN_FFN_DIM * BARUN_DIM;
    z.ffn_norm = p;
    return z;
}

int wubu_grow_insert_block(barun_model_t *m, int pos)
{
    if (!m || pos < 0 || pos > m->n_layers) return 0;
    if (m->n_layers >= BARUN_LAYERS) return 0;
    barun_block_t z = zero_block();
    if (!z.q_proj) return 0;
    /* NOTE: the displaced block at [n_layers] is overwritten by the
     * shift -- its buffers are the caller's ownership question (the CLI
     * runs short-lived; the test re-grows the same model, so a free here
     * would be a use-after-free for the test's owned buffers) */
    /* shift the blocks [pos..n) and their is_full rhythm up by one */
    for (int l = m->n_layers; l > pos; l--) {
        m->blocks[l] = m->blocks[l - 1];
        m->is_full[l] = m->is_full[l - 1];
        m->fire_sel[l] = m->fire_sel[l - 1];
    }
    m->blocks[pos] = z;
    /* the new zero block is an identity whatever its rhythm -- give it the
     * position's natural rhythm so the future growths stay consistent;
     * it never fires the residual selector (the identity blend is
     * harmless, but NOT firing keeps the selector order aligned) */
    m->is_full[pos] = ((pos + 1) % BARUN_FULL_EVERY == 0) ? 1 : 0;
    m->fire_sel[pos] = 0;
    m->n_layers++;
    return 1;
}

int wubu_grow_stack_block(barun_model_t *m, int src)
{
    if (!m || src < 0 || src >= m->n_layers) return 0;
    if (m->n_layers >= BARUN_LAYERS) return 0;
    barun_block_t z = zero_block();
    if (!z.q_proj) return 0;
    /* the G_stack copy: every weight buffer of the source block copied */
    barun_block_t *s = &m->blocks[src];
    size_t q = (size_t)BARUN_DIM * BARUN_HEADS * 64;
    size_t k = (size_t)BARUN_DIM * BARUN_KV_HEADS * 64;
    size_t f = (size_t)BARUN_DIM * BARUN_FFN_DIM * 2;
    size_t d = (size_t)BARUN_FFN_DIM * BARUN_DIM;
    memcpy(z.q_proj, s->q_proj, q * sizeof(float));
    memcpy(z.k_proj, s->k_proj, k * sizeof(float));
    memcpy(z.v_proj, s->v_proj, k * sizeof(float));
    memcpy(z.o_proj, s->o_proj, q * sizeof(float));
    memcpy(z.g_proj, s->g_proj, q * sizeof(float));
    memcpy(z.q_norm, s->q_norm, (size_t)BARUN_KV_HEADS * 64 * sizeof(float));
    memcpy(z.k_norm, s->k_norm, (size_t)BARUN_KV_HEADS * 64 * sizeof(float));
    memcpy(z.attn_norm, s->attn_norm, (size_t)BARUN_DIM * sizeof(float));
    memcpy(z.gate_up, s->gate_up, f * sizeof(float));
    memcpy(z.down, s->down, d * sizeof(float));
    memcpy(z.ffn_norm, s->ffn_norm, (size_t)BARUN_DIM * sizeof(float));
    m->blocks[m->n_layers] = z;
    m->is_full[m->n_layers] = ((m->n_layers + 1) % BARUN_FULL_EVERY == 0) ? 1 : 0;
    m->fire_sel[m->n_layers] = 0;
    m->n_layers++;
    return 1;
}

int wubu_grow_schedule(int t, int T, int base_layers, int max_layers,
                       float step_frac)
{
    if (T < 1 || t < 0) return base_layers < 1 ? 1 : base_layers;
    if (max_layers < base_layers) max_layers = base_layers;
    int growable = max_layers - base_layers;
    if (growable == 0 || step_frac <= 0) return base_layers;
    int steps_per = (int)(T * step_frac);
    if (steps_per < 1) steps_per = 1;
    int events = (int)(t / steps_per);
    if (events > growable) events = growable;
    return base_layers + events;
}

int wubu_grow_events(int T, int base_layers, int max_layers, float step_frac)
{
    if (T < 1 || max_layers <= base_layers || step_frac <= 0) return 0;
    int steps_per = (int)(T * step_frac);
    if (steps_per < 1) steps_per = 1;
    int events = T / steps_per;
    int growable = max_layers - base_layers;
    return events > growable ? growable : events;
}

int wubu_train_grow(barun_train_t *tr, int pos, int n_layers)
{
    if (!tr || pos < 0 || pos > n_layers || n_layers >= BARUN_LAYERS) return 0;
    /* the per-block pointer arrays: shift up then allocate the new slot */
#define SHIFT_ARR(ARR, SZ) do {                                        \
        if (ARR[n_layers]) free(ARR[n_layers]); /* the displaced unused */ \
        for (int l = n_layers; l > pos; l--) ARR[l] = ARR[l - 1];       \
        ARR[pos] = (float *)calloc((size_t)(SZ), sizeof(float));         \
        if (!ARR[pos]) return 0;                                         \
    } while (0)
    size_t q = (size_t)BARUN_DIM * BARUN_HEADS * 64;
    size_t k = (size_t)BARUN_DIM * BARUN_KV_HEADS * 64;
    size_t f = (size_t)BARUN_DIM * BARUN_FFN_DIM * 2;
    size_t d = (size_t)BARUN_FFN_DIM * BARUN_DIM;
    SHIFT_ARR(tr->q_proj_g, q); SHIFT_ARR(tr->k_proj_g, k);
    SHIFT_ARR(tr->v_proj_g, k); SHIFT_ARR(tr->o_proj_g, q);
    SHIFT_ARR(tr->g_proj_g, q); SHIFT_ARR(tr->gate_up_g, f);
    SHIFT_ARR(tr->down_g, d);
    SHIFT_ARR(tr->q_proj_m, q); SHIFT_ARR(tr->k_proj_m, k);
    SHIFT_ARR(tr->v_proj_m, k); SHIFT_ARR(tr->o_proj_m, q);
    SHIFT_ARR(tr->g_proj_m, q); SHIFT_ARR(tr->gate_up_m, f);
    SHIFT_ARR(tr->down_m, d);
#undef SHIFT_ARR
    /* the AdamW norm slots [4l+0..3]: free the displaced (the first
     * inactive block's slots), shift, then allocate the new block's */
    for (int k = 0; k < 4; k++) {
        free(tr->norm_g[4 * n_layers + k]);
        free(tr->norm_m[4 * n_layers + k]);
        free(tr->norm_v[4 * n_layers + k]);
    }
    for (int l = n_layers; l > pos; l--)
        for (int k = 0; k < 4; k++) {
            tr->norm_g[4 * l + k] = tr->norm_g[4 * (l - 1) + k];
            tr->norm_m[4 * l + k] = tr->norm_m[4 * (l - 1) + k];
            tr->norm_v[4 * l + k] = tr->norm_v[4 * (l - 1) + k];
        }
    int sz[4] = { BARUN_DIM, BARUN_DIM, 64, 64 };
    for (int k = 0; k < 4; k++) {
        tr->norm_g[4 * pos + k] = (float *)calloc((size_t)sz[k], sizeof(float));
        tr->norm_m[4 * pos + k] = (float *)calloc((size_t)sz[k], sizeof(float));
        tr->norm_v[4 * pos + k] = (float *)calloc((size_t)sz[k], sizeof(float));
        if (!tr->norm_g[4 * pos + k] || !tr->norm_m[4 * pos + k] ||
            !tr->norm_v[4 * pos + k]) return 0;
    }
    return 1;
}
