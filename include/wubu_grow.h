/* wubu_grow.h -- the model-growth operator (the Net2Net function-preserving
 * doctrine + the NeurIPS'24 growth-operator taxonomy + the Zhiqi Bu
 * progressive schedule). A zero-initialized block inserted into the
 * residual stack is an EXACT identity at init -- the insertion is
 * FUNCTION-PRESERVING, and that property is the DA oracle: the forward
 * before the growth must equal the forward after it. */
#ifndef WUBU_GROW_H
#define WUBU_GROW_H

#include "wubu.h"
#include "wubu_train.h"

/* Insert a NEW zero-initialized block at position pos (0 <= pos <=
 * m->n_layers). The existing blocks [pos..n) shift up, keeping their
 * weights and their is_full rhythm; the new block's weights are all
 * zero -- its residual branch contributes 0, so the model's function is
 * UNCHANGED (verifiable: run the forward before and after, compare).
 * Returns 1 on success, 0 when the model is already at BARUN_LAYERS. */
int wubu_grow_insert_block(wubu_model_t *m, int pos);

/* Stack a COPY of the block at src into a new block appended at the END
 * (the G_stack operator -- the "stacking saves 50% of the compute" recipe
 * from the NeurIPS'24 growth taxonomy). NOT function-preserving by
 * design; the gradients of the grown model must still be real (the FD
 * oracle checks the backward after the growth). Returns 1/0. */
int wubu_grow_stack_block(wubu_model_t *m, int src);

/* The Zhiqi Bu progressive schedule: the layer count to TRAIN at step t
 * of the total T steps, expanding by one layer every `step_frac` of the
 * horizon (Bu: every 10%; expanding at every 10% retains almost all the
 * performance). Monotonic; clamps to max_layers. */
int wubu_grow_schedule(int t, int T, int base_layers, int max_layers,
                       float step_frac);

/* The number of growth events the schedule fires over the horizon. */
int wubu_grow_events(int T, int base_layers, int max_layers, float step_frac);
/* Grow the train state to match a model that just grew: shift the
 * per-block gradient + momentum pointers [pos..n) up and allocate the
 * new block's (zero-initialized) buffers -- the grads and the Muon
 * momentum must follow the blocks the same way the weights did. */
int wubu_train_grow(wubu_train_t *tr, int pos, int n_layers);

#endif
