/* wubu_dsv4_layer.h — DeepSeek-V4 MLA attention bridge module.
 *
 * Bridges GGUF tensor loading (wubu_model.c) and the MLA forward pass
 * (wubu_mla.c). Self-contained C11, opaque struct, minimal includes.
 *
 * Not integrated into wubu_model.c (engine logic constraint) — this module
 * provides the MLA tensor name resolution and forward pass that wubu_model.c
 * can call when it detects a DSV4 layer.
 */
#ifndef WUBU_DSV4_LAYER_H
#define WUBU_DSV4_LAYER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque DSV4 layer handle (wraps wubu_mla_t + weight pointers) */
typedef struct wubu_dsv4_layer wubu_dsv4_layer_t;

/* Create a DSV4 layer context with MLA dims.
 * Returns NULL on bad args or OOM. */
wubu_dsv4_layer_t *wubu_dsv4_layer_create(int hidden_dim, int n_heads,
                                           int head_dim, int q_lora_rank,
                                           int kv_lora_rank, int rope_head_dim);

/* Destroy. NULL-safe. */
void wubu_dsv4_layer_free(wubu_dsv4_layer_t *dl);

/* Set the weight pointers for this layer's MLA tensors.
 * Called by wubu_model.c after gguf_find_tensor resolves the blob addresses.
 * Returns 1 on success, -1 on error. */
int wubu_dsv4_layer_load_tensors(wubu_dsv4_layer_t *dl, int layer_idx,
                                  const float *W_DQ, const float *W_UQ,
                                  const float *W_DKV, const float *W_UK,
                                  const float *W_UV, const float *W_O,
                                  const float *attn_norm);

/* Forward pass for a single token using DSV4 MLA attention.
 * Returns 0 on success, -1 on error. */
int wubu_dsv4_layer_forward(const wubu_dsv4_layer_t *dl,
                             const float *x,
                             const float *kv_cache,
                             int pos,
                             float *out);

/* Resolve DSV4 MLA tensor names for GGUF lookup.
 * tensor_type: "q_a", "q_b", "kv", "k_up", "v_up", "o_a", "o_b", "norm"
 * Returns a malloc'd string (caller frees), or NULL if type is unknown. */
char *wubu_dsv4_tensor_name(int layer, const char *tensor_type);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_DSV4_LAYER_H */
