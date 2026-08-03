/* wubu_masked_ce.h -- the masked next-token cross-entropy: the loss is
 * computed ONLY on the assistant tokens (the Hermes 69%-output-token +
 * Orchard obs-masking doctrine, now as the actual training loss). The
 * wubu_traj_sft per-segment masks feed this loss directly. FD-verifiable:
 * the masked positions' gradients must be zero and the unmasked ones must
 * match the finite differences. */
#ifndef WUBU_MASKED_CE_H
#define WUBU_MASKED_CE_H

#include <stdint.h>

/* logits [seq*vocab]: the model's logits (row s = position s).
 * tokens [seq]: the targets (position s predicts tokens[s]).
 * mask [seq]: 1 = the position trains, 0 = masked (obs/user/context).
 * loss (out): the mean-reduced masked CE; grad [seq*vocab] (out, may be
 *   NULL): dL/dlogits. Returns 1 on success. */
int wubu_masked_ce(const float *logits, const uint16_t *tokens,
                   const float *mask, int seq, int vocab,
                   float *loss, float *grad);

/* The effective training-token fraction (the masked count / seq). */
float wubu_masked_ce_frac(const float *mask, int seq);

#endif
