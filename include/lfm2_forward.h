#ifndef LFM2_FORWARD_H
#define LFM2_FORWARD_H

#include "wubu_lfm2.h"

#ifdef __cplusplus
extern "C" {
#endif

/* LFM2.5 forward orchestrator (C11, self-contained).
 * Owns the residual loop and dispatches to the conv/attn/ffn op modules.
 * emb[B*T*d_model] in -> logits[vocab] out (last token). */

#ifdef __cplusplus
}
#endif

#endif /* LFM2_FORWARD_H */
