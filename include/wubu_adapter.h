/* wubu_adapter.h — hot-swappable AGI component adapter framework
 *
 * The AGI is built from interchangeable parts: attention mechanisms,
 * feedforward blocks, layer norms, activation functions, tokenizers.
 * Each part is an "adapter" — an opaque struct with a dispatch table
 * of function pointers. Parts can be swapped at compile time (via
 * #define WUBU_ADAPTER_<NAME>) or at runtime (via registration).
 *
 * This is the C11 equivalent of nn.ModuleDict / plugin system:
 *
 *   wubu_adapter_t *attn = wubu_adapter_lookup("attn.local");
 *   attn->ops->forward(attn, ctx, Q, K, V, out);
 *
 * Design: the same pattern as wubu_kernel.h (dispatch table) applied
 * to AGI architecture components instead of hardware kernels.
 *
 * Frame: wubu_adapter.h — the opaque type + dispatch vtable
 * Buffer: wubu_adapter_registry.c — the slot map (name → adapter)
 * Shims:  wubu_adapter_compat.c — backward-compat bridges (v1↔v2)
 *
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_ADAPTER_H
#define WUBU_ADAPTER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Component types (what can be swapped) ---- */

typedef enum {
    WUBU_COMP_ATTN = 0,     /* attention mechanism */
    WUBU_COMP_FFN  = 1,     /* feedforward network */
    WUBU_COMP_NORM = 2,     /* layer norm */
    WUBU_COMP_ACT  = 3,     /* activation function */
    WUBU_COMP_TOK  = 4,     /* tokenizer */
    WUBU_COMP_LR   = 5,     /* learning rate schedule */
    WUBU_COMP_OPT  = 6,     /* optimizer */
    WUBU_COMP_QUANT= 7,     /* quantization strategy */
    WUBU_COMP_KVC  = 8,     /* KV cache policy */
    WUBU_COMP_MAX  = 9
} wubu_component_type_t;

/* Generic context for forward/backward (carries KV, params, etc.) */
typedef struct wubu_adapter_ctx wubu_adapter_ctx_t;

/* Per-component forward signature.
 * ctx carries the model state, x is input, out is output.
 * Returns 0 on success. */
typedef int (*wubu_adapter_forward_fn)(void *self,
                                        wubu_adapter_ctx_t *ctx,
                                        const float *x, size_t n_in,
                                        float *out, size_t n_out);

typedef int (*wubu_adapter_backward_fn)(void *self,
                                         wubu_adapter_ctx_t *ctx,
                                         const float *grad_out, size_t n_out,
                                         float *grad_in, size_t n_in);

typedef void (*wubu_adapter_init_fn)(void *self);
typedef void (*wubu_adapter_free_fn)(void *self);

/* The dispatch vtable — one per component instance.
 * This is the "frame" that every adapter plugs into. */
typedef struct {
    wubu_component_type_t type;
    const char *name;        /* e.g. "attn.local_window", "ffn.gated_swiglu" */
    const char *version;    /* semantic version for compat checks */

    wubu_adapter_init_fn   init;
    wubu_adapter_free_fn   free_fn;
    wubu_adapter_forward_fn forward;
    wubu_adapter_backward_fn backward;
} wubu_adapter_ops_t;

/* The adapter itself — opaque to everyone except its own module.
 * The first field is ALWAYS the ops pointer (like C++ vtables). */
struct wubu_adapter {
    const wubu_adapter_ops_t *ops;
    /* module-specific state follows */
};

typedef struct wubu_adapter wubu_adapter_t;

/* ---- Core API ---- */

/* Register a named adapter (makes it swappable). Returns 0 on success. */
int wubu_adapter_register(wubu_adapter_t *adapter);

/* Look up an adapter by name (e.g. "attn.local_window").
 * Returns NULL if not found. */
wubu_adapter_t *wubu_adapter_lookup(const char *name);

/* Look up by component type + name. */
wubu_adapter_t *wubu_adapter_lookup_type(wubu_component_type_t type,
                                          const char *name);

/* List all registered adapters of a given type.
 * Returns the count, fills names[] with up to n_names entries. */
int wubu_adapter_list(wubu_component_type_t type,
                       char names[][64], int n_names);

/* Replace a running adapter with a new one (hot-swap).
 * The old adapter is freed via its free_fn.
 * Returns 0 on success, -1 if name not found. */
int wubu_adapter_swap(const char *name, wubu_adapter_t *new_adapter);

/* Get the current adapter for a component type + name.
 * This is the hot path — inline-friendly. */
wubu_adapter_t *wubu_adapter_current(wubu_component_type_t type,
                                      const char *name);

/* Check if an adapter is compatible with a target version string.
 * Returns 1 if compatible, 0 if not. */
int wubu_adapter_compat(const wubu_adapter_ops_t *ops,
                         const char *target_version);

/* Free all adapters (called at shutdown). */
void wubu_adapter_shutdown(void);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_ADAPTER_H */
