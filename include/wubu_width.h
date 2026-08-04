/* wubu_width.h -- the width expansion (Net2Net dynamic-dims refactor):
 * grow the hidden width of an existing model by a factor of 2,
 * using the ZERO-PADDING identity: every expanded weight keeps the old
 * block in its top-left corner EXACTLY and zeroes the new rows and
 * columns; the embedding's right half is zero; the norms' new half is
 * at the identity scale (1.0).
 *
 * Function-preserving claim (the zero-padding math): with the hidden
 * stream x' = [x; 0] (the new half zero), any expanded matrix
 * W' = [[W, 0], [0, 0]] gives W'x' = [Wx; 0] -- the left half is the
 * OLD output exactly, the right half stays zero, so a stack of
 * expanded blocks computes the identical left-half stream. (The
 * attention's new heads have zero q/k/v, so they contribute zero
 * through the o_proj's zeroed new columns.)
 *
 * The engine-side threading (wubu_buf_t, forward, backprop, train
 * state, checkpoint format all carry the fixed WUBU_DIM) is the
 * dynamic-dims refactor; this module produces the weight-level
 * expansion that refactor consumes. */
#ifndef WUBU_WIDTH_H
#define WUBU_WIDTH_H

#include "wubu.h"

/* Expand the model's hidden width by doubling it (WUBU_DIM *= 2).
 * Returns 1 on success, 0 on allocation failure.
 * Expands: every block's weights (attn q/k/v/o/g, gate_up, down,
 * norms), the embedding (right half zero), the final norm, and the
 * selectors. The old weights are preserved EXACTLY (no scaling). */
int wubu_width_expand(wubu_model_t *m);

#endif