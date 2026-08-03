/*
 * wubu_barun_train.h -- the BarunLM training core (the AGI brain-cluster loop).
 *
 * The wizard is an INFERENCE engine today; this module makes it a
 * TRAINING engine too. BarunLM-35M is the mustard seed: the training
 * loop grows it -- more tokens from the research repos, more
 * parameters, more knowledge -- all designed and trained in-house.
 *
 * The reference recipe (reproduced faithfully):
 *   - Muon optimizer (the reference used it for the final 4B tokens:
 *     peak lr 1e-4, weight decay 0.1, batch 48, seq 2048)
 *   - cross-entropy on the next token
 *   - the hybrid local/global attention forward (wubu_barun)
 *   - weight-decay applied via the Muon Newton-Schulz style update
 *
 * Training is memory-bound: we accumulate gradients over micro-batches
 * and update with Muon. Pure C11, no third-party deps.
 */
#ifndef WUBU_BARUN_TRAIN_H
#define WUBU_BARUN_TRAIN_H

#include "wubu_barun.h"

/* The training config (the reference recipe). */
typedef struct {
    float  lr;             /* 1e-4 peak */
    float  weight_decay;   /* 0.1 */
    float  grad_clip;      /* 1.0 */
    uint32_t batch_size;   /* 48 (reference) -- we micro-batch */
    uint32_t seq_len;      /* 2048 */
    uint32_t warmup_steps;
    uint32_t max_steps;
    float  muon_momentum;  /* 0.95 */
    int    adam_for_embed; /* the embedding + norms use AdamW, the
                              matrices use Muon (reference split) */
} barun_train_cfg_t;

/* Gradient accumulators: one float per weight, only for the Muon-updated
 * matrices (the big ones). The norms + embeddings use AdamW states. */
typedef struct {
    /* per-block matrix gradients */
    float *q_proj_g[BARUN_LAYERS];  /* [448,448] */
    float *k_proj_g[BARUN_LAYERS];  /* [448,64] */
    float *v_proj_g[BARUN_LAYERS];
    float *o_proj_g[BARUN_LAYERS];
    float *g_proj_g[BARUN_LAYERS];
    float *gate_up_g[BARUN_LAYERS]; /* [448,2456] */
    float *down_g[BARUN_LAYERS];    /* [1228,448] */
    float *selectors_g[BARUN_SELECTORS];
    /* AdamW states for the embedding */
    float *emb_g;   /* [16384,448] the gradient accumulator */
    float *emb_m;   /* [16384,448] the AdamW first moment */
    float *emb_v;   /* [16384,448] the AdamW second moment */
    /* AdamW states for the norms (small) */
    float *norm_m[BARUN_LAYERS * 3 + 1];
    float *norm_v[BARUN_LAYERS * 3 + 1];
    /* Muon states: the momentum per matrix (Newton-Schulz iteration) */
    float *q_proj_m[BARUN_LAYERS];
    float *k_proj_m[BARUN_LAYERS];
    float *v_proj_m[BARUN_LAYERS];
    float *o_proj_m[BARUN_LAYERS];
    float *g_proj_m[BARUN_LAYERS];
    float *gate_up_m[BARUN_LAYERS];
    float *down_m[BARUN_LAYERS];
    float *selectors_m[BARUN_SELECTORS];
    /* telemetry */
    uint32_t micro_steps;
    double   grad_norm_sum;
    double   loss_sum;
} barun_train_t;

/* T1: initialize the training state (allocates the gradient buffers). */
int barun_train_init(barun_train_t *tr, const barun_model_t *m);

/* T2: zero the accumulated gradients. */
int barun_train_zero_grad(barun_train_t *tr);

/* T3: accumulate the gradient of one micro-batch. The caller provides
 * the forward activations; the trainer computes the next-token CE
 * gradient and backprops through the last layer only (the reference
 * trains with the full backprop; the C11 trainer currently does the
 * analytic last-layer + embedding gradient, with the hidden layers'
 * grads accumulated numerically per micro-batch). Returns the loss. */
float barun_train_microbatch(barun_model_t *m, barun_train_t *tr,
                             barun_buf_t *b, const uint16_t *tokens,
                             size_t n_tokens);

/* T4: the Muon+AdamW optimizer step. */
int barun_train_step(barun_model_t *m, barun_train_t *tr,
                     const barun_train_cfg_t *cfg, uint32_t step);

/* T5: the full training loop -- one step = batch of micro-batches. */
float barun_train_step_loop(barun_model_t *m, barun_train_t *tr,
                            barun_buf_t *b, const uint16_t *tokens,
                            size_t n_tokens, const barun_train_cfg_t *cfg,
                            uint32_t step);

/* T6: the learning-rate schedule (warmup + cosine decay). */
float barun_train_lr(const barun_train_cfg_t *cfg, uint32_t step);

/* T7: free the training state. */
void barun_train_free(barun_train_t *tr);

#endif