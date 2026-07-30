#ifndef WUBU_SSM_WORKSPACE_H
#define WUBU_SSM_WORKSPACE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * wubu_ssm_workspace — per-SSM-layer static workspace pool.
 *
 * Instead of malloc/free on every wubu_ssm_forward() call, each layer
 * pre-allocates scratch buffers once at wubu_ssm_workspace_init() time
 * and reuses them for every decode step.
 *
 * For Qwen3.6-27B d_model=5120 CONV_DIM=8192 SSM_V_HEADS=48 SSM_D_STATE=128:
 *   One layer workspace ≈ 10 MB
 *   Full model (64 layers) ≈ 640 MB — fine on WSL 13 GB host.
 *
 * The pool is single-threaded; multi-threaded callers must serialize
 * the forward call per layer or pass a per-layer pool from TLS.
 */

#define WUBU_SSM_MAX_LAYERS 128

typedef struct {
    float *qkv_all;
    float *z_all;
    float *beta_raw;
    float *alpha_raw;
    float *conv_input;
    float *conv_output;
    float *q_conv;
    float *k_conv;
    float *v_conv;
    float *q_norm;
    float *k_norm;
    float *delta_out;
    float *z_silu;
} wubu_ssm_workspace_t;

int  wubu_ssm_workspace_init(int max_layers, int B, int T);
void wubu_ssm_workspace_shutdown(void);
wubu_ssm_workspace_t *wubu_ssm_workspace_get(int layer_idx);

#endif /* WUBU_SSM_WORKSPACE_H */
