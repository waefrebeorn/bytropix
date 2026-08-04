/*
 * wubu_ssm_workspace.c — per-SSM-layer scratch pool.
 *
 * Replaces the per-call 13xmalloc + 13xfree inside wubu_ssm_forward().
 *
 * Buffers are sized to the MAX sequence length (g_cur_T) ever requested, and
 * grown on demand. This fixes a SIGSEGV: the old code sized buffers once at
 * init() time (g_T) and never reallocated, so a later forward with a larger T
 * (e.g. prefill after a T=1 decode init) overran the cached conv_input.
 */

#include "wubu_ssm_workspace.h"
#include "wubu_ssm.h"
#include <stdlib.h>
#include <string.h>

#define WUBU_SSM_WORKSPACE_MAX_LAYERS 128

static wubu_ssm_workspace_t g_pool[WUBU_SSM_WORKSPACE_MAX_LAYERS];
static int g_initialized = 0;
static int g_max_layers = 0;
static int g_B = 0;
static int g_T = 0;      /* init-time T (reference only) */
static int g_cur_T = 0;  /* current active sequence length (max seen) */
static int g_cap_T = 0;  /* T the pooled buffers are currently sized for */

static inline float *ws_alloc(size_t n) {
    return (float *)calloc(n, sizeof(float));
}

int wubu_ssm_workspace_init(int max_layers, int B, int T) {
    if (g_initialized) return 0;
    if (max_layers > WUBU_SSM_WORKSPACE_MAX_LAYERS) max_layers = WUBU_SSM_WORKSPACE_MAX_LAYERS;
    g_max_layers = max_layers;
    g_B = B;
    g_T = T;
    g_cur_T = T;
    g_cap_T = 0;   /* force first _get to size to g_cur_T */
    memset(g_pool, 0, sizeof(g_pool));
    g_initialized = 1;
    return 0;
}

/* Update the active sequence length. Buffers grow to the MAX T ever seen so a
   later larger-T forward (e.g. prefill after a T=1 decode init) never overruns
   the cached scratch. Fixes SIGSEGV in wubu_ssm_forward conv_input copy. */
void wubu_ssm_workspace_set_t(int T) {
    if (T > g_cur_T) g_cur_T = T;
}

void wubu_ssm_workspace_shutdown(void) {
    if (!g_initialized) return;
    for (int i = 0; i < g_max_layers; i++) {
        free(g_pool[i].qkv_all);     g_pool[i].qkv_all = NULL;
        free(g_pool[i].z_all);       g_pool[i].z_all = NULL;
        free(g_pool[i].beta_raw);    g_pool[i].beta_raw = NULL;
        free(g_pool[i].alpha_raw);   g_pool[i].alpha_raw = NULL;
        free(g_pool[i].conv_input);  g_pool[i].conv_input = NULL;
        free(g_pool[i].conv_output); g_pool[i].conv_output = NULL;
        free(g_pool[i].q_conv);      g_pool[i].q_conv = NULL;
        free(g_pool[i].k_conv);      g_pool[i].k_conv = NULL;
        free(g_pool[i].v_conv);      g_pool[i].v_conv = NULL;
        free(g_pool[i].q_norm);      g_pool[i].q_norm = NULL;
        free(g_pool[i].k_norm);      g_pool[i].k_norm = NULL;
        free(g_pool[i].delta_out);   g_pool[i].delta_out = NULL;
        free(g_pool[i].z_silu);      g_pool[i].z_silu = NULL;
    }
    g_initialized = 0;
}

wubu_ssm_workspace_t *wubu_ssm_workspace_get(int layer_idx) {
    if (!g_initialized || layer_idx < 0 || layer_idx >= g_max_layers) return NULL;
    wubu_ssm_workspace_t *p = &g_pool[layer_idx];

    /* Grow all buffers if the current T exceeds what we are sized for. Free the
       old (smaller) buffers so they are reallocated below at the larger T. */
    if (g_cur_T > g_cap_T) {
        free(p->qkv_all);     p->qkv_all = NULL;
        free(p->z_all);       p->z_all = NULL;
        free(p->beta_raw);    p->beta_raw = NULL;
        free(p->alpha_raw);   p->alpha_raw = NULL;
        free(p->conv_input);  p->conv_input = NULL;
        free(p->conv_output); p->conv_output = NULL;
        free(p->q_conv);      p->q_conv = NULL;
        free(p->k_conv);      p->k_conv = NULL;
        free(p->v_conv);      p->v_conv = NULL;
        free(p->q_norm);      p->q_norm = NULL;
        free(p->k_norm);      p->k_norm = NULL;
        free(p->delta_out);   p->delta_out = NULL;
        free(p->z_silu);      p->z_silu = NULL;
        g_cap_T = g_cur_T;
    }

    const int N = g_B * g_cur_T;
    const int C = CONV_DIM;
    if (!p->qkv_all)    p->qkv_all    = ws_alloc((size_t)N * (KEY_DIM * 2 + VALUE_DIM));
    if (!p->z_all)      p->z_all      = ws_alloc((size_t)N * VALUE_DIM);
    if (!p->beta_raw)   p->beta_raw   = ws_alloc((size_t)N * DT_RANK);
    if (!p->alpha_raw)  p->alpha_raw  = ws_alloc((size_t)N * DT_RANK);
    if (!p->conv_input) p->conv_input = ws_alloc((size_t)g_B * (g_cur_T + CONV_KERNEL - 1) * C);
    if (!p->conv_output)p->conv_output= ws_alloc((size_t)N * C);
    if (!p->q_conv)     p->q_conv     = ws_alloc((size_t)N * KEY_DIM);
    if (!p->k_conv)     p->k_conv     = ws_alloc((size_t)N * KEY_DIM);
    if (!p->v_conv)     p->v_conv     = ws_alloc((size_t)N * VALUE_DIM);
    if (!p->q_norm)     p->q_norm     = ws_alloc((size_t)N * KEY_DIM);
    if (!p->k_norm)     p->k_norm     = ws_alloc((size_t)N * KEY_DIM);
    if (!p->delta_out)  p->delta_out  = ws_alloc((size_t)N * VALUE_DIM);
    if (!p->z_silu)     p->z_silu     = ws_alloc((size_t)N * VALUE_DIM);
    return p;
}
