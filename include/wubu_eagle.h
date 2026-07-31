#ifndef WUBU_EAGLE_H
#define WUBU_EAGLE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * wubu_eagle.h — EAGLE self-draft speculative decoding (G01).
 *
 * EAGLE: Speculative Decoding with Small Draft Models
 * (Zhang et al., ICLR 2024, 2402.00366).
 *
 * Uses a small draft model (truncated target) to generate K tokens
 * in parallel, then verifies them with the full target model in a
 * single batched forward pass. Accepted tokens are kept; rejected
 * tokens are re-generated.
 *
 * This is the CPU-only implementation for WuBuOS. No third-party deps.
 */

#include "wubu_model.h"

typedef struct {
    wubu_model_t *target;   /* pointer to the large target model */
    int draft_layers;       /* number of layers in draft model (truncated) */
} wubu_eagle_draft_t;

/* Initialize a draft model from a target model by truncating layers.
 * Returns 0 on success, -1 on failure. */
int wubu_eagle_draft_init(wubu_eagle_draft_t *draft, wubu_model_t *target,
                          int draft_layers);

/* Generate K speculative tokens using the draft model (truncated forward).
 * draft_tokens: output array of K token IDs.
 * Returns number of tokens generated (0 on error). */
int wubu_eagle_draft_generate(wubu_eagle_draft_t *draft,
                               const int *prompt, int prompt_len,
                               int *draft_tokens, int max_draft);

/* Verify draft tokens against the target model.
 * accepted_tokens: output array of verified token IDs.
 * Returns number of accepted tokens (>= 0). */
int wubu_eagle_verify(wubu_model_t *target,
                      const int *prompt, int prompt_len,
                      const int *draft_tokens, int num_draft,
                      int *accepted_tokens, int max_accepted);

/* Full EAGLE speculative decode: draft + verify + re-generate.
 * Returns total accepted tokens. */
int wubu_eagle_speculative_decode(wubu_eagle_draft_t *draft,
                                  wubu_model_t *target,
                                  const int *prompt, int prompt_len,
                                  int *output_tokens, int max_output);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_EAGLE_H */