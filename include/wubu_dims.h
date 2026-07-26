/*
 * wubu_dims.h -- runtime model dimensions for bytropix.
 *
 * bytropix's SSM/GQA/MoE forward passes were originally hardcoded to a
 * single model via compile-time macros (D_MODEL=2048, CONV_DIM=8192,
 * ...). That is broken: it cannot load the real Colonel models
 * (Qwen3.6-27B hidden=5120, Agents-A1-4B hidden=2560, ...).
 *
 * This module makes those macros read from a single runtime global
 * WUBU_DIMS, which the loader sets from the ACTUAL tensor shapes of the
 * checkpoint being loaded. Every forward function then runs at the
 * model's real dimensions -- no edits inside the 2900-line forward
 * bodies, just a header redefinition. One source of truth, split out so
 * no other file is a "god header".
 *
 * C11, opaque-ish: only the struct + the macro aliases live here.
 */
#ifndef WUBU_DIMS_H
#define WUBU_DIMS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int d_model;       /* hidden size                       */
    int conv_dim;     /* SSM in_proj_qkv out dim (Q+K+V)   */
    int value_dim;    /* SSM in_proj_z out (VALUE_DIM)     */
    int key_dim;      /* SSM Q/K dim (K heads)             */
    int dt_rank;      /* ssm_time_step_rank                */
    int ssm_d_state;  /* SSM state dim (head_k = head_v)   */
    int ssm_k_heads;  /* SSM num key heads                 */
    int ssm_v_heads;  /* SSM num value heads               */
    int conv_kernel;  /* conv1d kernel size                */
    int gqa_q_heads;  /* attention query heads             */
    int gqa_kv_heads; /* attention kv heads (GQA)          */
    int gqa_head_dim; /* attention head dim                */
    int gqa_kv_dim;   /* kv_heads * head_dim               */
} wubu_dims_t;

/* The single active dimension set. The loader calls wubu_dims_set()
 * right before loading/running a model. */
extern wubu_dims_t WUBU_DIMS;

/* Set dims explicitly (loader computes them from real tensor shapes). */
void wubu_dims_set(const wubu_dims_t *d);

/* Recompute derived fields (conv_dim, key_dim, gqa_kv_dim) from the
 * primary fields already present in *d. Safe to call after a partial set. */
void wubu_dims_finalize(wubu_dims_t *d);

/* Defaults matching bytropix's original 2048-hidden build (GGUF path
 * stays byte-compatible). Call before any legacy GGUF load. */
void wubu_dims_default(void);

#ifdef __cplusplus
}
#endif

/* ---- Macro aliases: every forward reads these, now runtime ---- */
#define D_MODEL       WUBU_DIMS.d_model
#define CONV_DIM      WUBU_DIMS.conv_dim
#define VALUE_DIM     WUBU_DIMS.value_dim
#define KEY_DIM       WUBU_DIMS.key_dim
#define DT_RANK       WUBU_DIMS.dt_rank
#define SSM_D_STATE   WUBU_DIMS.ssm_d_state
#define SSM_K_HEADS   WUBU_DIMS.ssm_k_heads
#define SSM_V_HEADS   WUBU_DIMS.ssm_v_heads
#define CONV_KERNEL   WUBU_DIMS.conv_kernel
#define GQA_Q_HEADS   WUBU_DIMS.gqa_q_heads
#define GQA_KV_HEADS  WUBU_DIMS.gqa_kv_heads
#define GQA_HEAD_DIM  WUBU_DIMS.gqa_head_dim
#define GQA_KV_DIM    WUBU_DIMS.gqa_kv_dim

#endif /* WUBU_DIMS_H */
