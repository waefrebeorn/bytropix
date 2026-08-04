/*
 * wubu_dims.c -- runtime dimension global for wubuwizard.
 * See wubu_dims.h. The loader sets WUBU_DIMS from real tensor shapes;
 * the forward passes read it via the macro aliases.
 */
#include "wubu_dims.h"

/* CPU-only stub for the GPU mirror call. When GPU_SUPPORT is defined the real
 * definition lives in wubu_dims_gpu.cu (which mirrors to a CUDA __constant__).
 * On a CPU/Windows build there is no device to mirror to, so this is a no-op. */
#ifndef GPU_SUPPORT
void wubu_dims_sync_gpu(void) { /* no GPU: nothing to mirror */ }
#endif

wubu_dims_t WUBU_DIMS = {0};

void wubu_dims_set(const wubu_dims_t *d) {
    if (!d) return;
    WUBU_DIMS = *d;
    wubu_dims_finalize(&WUBU_DIMS);
    wubu_dims_sync_gpu();   /* mirror to CUDA __constant__ for device kernels */
}

void wubu_dims_finalize(wubu_dims_t *d) {
    if (!d) return;
    /* Derive the cross-products the forward expects. */
    if (d->ssm_k_heads > 0 && d->ssm_d_state > 0)
        d->key_dim = d->ssm_d_state * d->ssm_k_heads;
    if (d->ssm_k_heads > 0 && d->ssm_v_heads > 0 && d->ssm_d_state > 0)
        d->conv_dim = d->key_dim * 2 + (d->ssm_d_state * d->ssm_v_heads);
    if (d->gqa_kv_heads > 0 && d->gqa_head_dim > 0)
        d->gqa_kv_dim = d->gqa_kv_heads * d->gqa_head_dim;
}

void wubu_dims_default(void) {
    wubu_dims_t d;
    d.d_model     = 2048;
    d.ssm_d_state = 128;
    d.ssm_k_heads = 16;
    d.ssm_v_heads = 32;
    d.conv_kernel = 4;
    d.dt_rank     = 32;
    d.gqa_q_heads  = 16;
    d.gqa_kv_heads = 2;
    d.gqa_head_dim = 256;
    /* conv_dim / key_dim / gqa_kv_dim derived: */
    d.key_dim = d.ssm_d_state * d.ssm_k_heads;            /* 2048 */
    d.conv_dim = d.key_dim * 2 + (d.ssm_d_state * d.ssm_v_heads); /* 8192 */
    d.gqa_kv_dim = d.gqa_kv_heads * d.gqa_head_dim;       /* 512  */
    d.value_dim = d.ssm_d_state * d.ssm_v_heads;          /* 4096 */
    WUBU_DIMS = d;
    wubu_dims_sync_gpu();
}
