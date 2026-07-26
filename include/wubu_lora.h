#ifndef WUBU_LORA_H
#define WUBU_LORA_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * wubu_lora.h -- LoRA (Low-Rank Adaptation) application.
 *
 * Needed for BTL-3: a rank-32 / alpha-64 PEFT adapter on top of the
 * Qwen3.6-27B base. Applies  deltaW = (alpha / rank) * B @ A
 * to a target linear module's weight matrix, in place:
 *
 *     W' = W + (alpha/rank) * (B [r x in]) @ (A [r x out])^T
 *        = W + scale * B^T @ A        (depending on storage layout)
 *
 * Storage follows the standard PEFT layout used by the BTL-3 release:
 *   lora_A.weight : [rank, in_features]   (down-projection)
 *   lora_B.weight : [out_features, rank]  (up-projection)
 *   merged: y = x @ (W^T + scale * B^T @ A)  ==  (x@W^T) + scale*(x@A^T)@B^T
 *
 * The module this LoRA attaches to is opaque to callers.
 */

typedef struct wubu_lora wubu_lora_t;

// Create a LoRA adapter for one module.
//   rank, alpha : LoRA hyperparameters (e.g. 32, 64).
//   in_f, out_f : linear module dimensions.
// Returns NULL on failure.
wubu_lora_t *wubu_lora_create(int rank, float alpha,
                                int in_f, int out_f);

// Free.
void wubu_lora_free(wubu_lora_t *l);

// Load A/B f32 weights from caller buffers (row-major:
//   A = [rank, in_f], B = [out_f, rank]). Copies internally.
// Returns 0 on success, -1 on dimension mismatch.
int wubu_lora_load_f32(wubu_lora_t *l,
                         const float *A, const float *B);

// Load A/B directly from safetensors raw f32 buffers.
int wubu_lora_load_raw(wubu_lora_t *l,
                         const float *A, const float *B);

// Apply the LoRA delta to a base weight matrix W [out_f, in_f] IN PLACE.
// W is row-major (output-major).  y = W + scale * (B^T @ A)
// Returns 0 on success.
int wubu_lora_apply(const wubu_lora_t *l, float *W);

// Compute the LoRA contribution x @ (scale * B^T @ A) for one input
// vector x [in_f] -> out [out_f]. Writes to `out`. Used when the
// base weight stays frozen and the delta is applied at inference time.
int wubu_lora_forward(const wubu_lora_t *l,
                        const float *x, float *out);

// Effective scale (alpha / rank). Exposed for diagnostics/tests.
float wubu_lora_scale(const wubu_lora_t *l);

#ifdef __cplusplus
}
#endif

#endif // WUBU_LORA_H
