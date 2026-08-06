// include/wubu_graph.h
// The universal graph IR — the mathematical-engine subtext (AN19, step 1).
//
// Every model — LLM, music (MusicGen), image diffusion (DiT), video (Sora),
// GANs, encoders/decoders — is the same thing at this layer:
//
//     a directed graph of typed tensor operations.
//
//   * TENSOR NODES carry a wubu_weight_t (data + GGML type tag + count) and
//     a logical shape. A weight tensor's data is storage-backed (mmap'd or
//     heap); an activation tensor's data is a runtime buffer owned by the
//     executor.
//   * OP NODES reference input tensors and one output tensor, and dispatch
//     through a REGISTRY (function-pointer table, like wubu_kernel) — so
//     adding an op kind (conv, STFT, cross-attention) is a table entry,
//     never an engine branch.
//   * The EXECUTOR walks ops in topological order, allocating activation
//     buffers, materializing weights through the universal materializer
//     (wubu_weight_to_f32 / wubu_weight_matmul), and copying results out.
//
// Binders (GGUF/ONNX/safetensors importers) emit graphs; the old
// per-architecture forwards (wubu_ssm_forward, wubu_gqa_forward, ...)
// become OPTIMIZED SUBGRAPH EXECUTORS the graph IR can pattern-match.
// There is no per-model struct to keep in sync — a weight is one
// descriptor, one type tag, one consumer. The _q/_raw/_f32 triplication
// is impossible here by construction.
//
// Design doc: docs/universal-manifold-design.md (AN19).
#ifndef WUBU_GRAPH_H
#define WUBU_GRAPH_H

#include <stdint.h>
#include <stddef.h>
#include "wubu_weight.h"   // wubu_weight_t — the universal descriptor

#ifdef __cplusplus
extern "C" {
#endif

/* ---- op kinds ------------------------------------------------------- */
/* The built-in lattice. Binders and future modalities register more via
 * wubu_graph_register_op. These are the ops the executor knows natively. */
typedef enum {
    WUBU_OP_NONE = 0,
    WUBU_OP_MATMUL,   /* y = x @ W ;  x:[k] act, W:[n,k] weight, y:[n]      */
    WUBU_OP_RELU,     /* y = max(x, 0) elementwise                          */
    WUBU_OP_ADD,      /* y = a + b elementwise (same element count)         */
    WUBU_OP_SILU,     /* y = x * sigmoid(x) elementwise                     */
    WUBU_OP_COUNT
} wubu_op_kind_t;

/* ---- opaque graph --------------------------------------------------- */
typedef struct wubu_graph wubu_graph_t;

/* ---- tensor & op node views (read-only) ----------------------------- */
typedef struct {
    wubu_weight_t w;     /* descriptor; .data == NULL for activation nodes */
    char          name[64];
    int64_t       dims[4];
    int           n_dims;
    int           is_weight;   /* 1 = weight (storage-backed), 0 = activation */
} wubu_graph_tensor_t;

typedef struct {
    int   kind;          /* wubu_op_kind_t or a registered extension kind */
    int   inputs[8];     /* tensor node indices                          */
    int   n_inputs;
    int   output;        /* tensor node index (must differ from inputs)  */
} wubu_graph_op_t;

/* ---- construction --------------------------------------------------- */
wubu_graph_t *wubu_graph_create(void);
void          wubu_graph_free(wubu_graph_t *g);

/* Add a tensor node. If w != NULL and w->data != NULL it is a WEIGHT node
 * (storage-backed, materialized by the executor on first use); otherwise
 * it is an ACTIVATION node (buffer owned by the executor, fed or written
 * by ops). dims[] gives the logical shape (dims[0] = fastest-varying).
 * Returns the tensor node index, or -1 on error. */
int wubu_graph_add_tensor(wubu_graph_t *g, const char *name,
                          const wubu_weight_t *w,
                          const int64_t *dims, int n_dims);

/* Add an op node. inputs[] are tensor indices (already added), output is
 * the tensor index the op writes. Returns the op node index, or -1. */
int wubu_graph_add_op(wubu_graph_t *g, int kind,
                      const int *inputs, int n_inputs, int output);

/* Validate the graph: every op's input tensor must be a weight, a feed
 * placeholder, or the output of an EARLIER op (topological order).
 * Returns 0 on success, -1 with the offending op index via *bad_op. */
int wubu_graph_validate(const wubu_graph_t *g, int *bad_op);

/* ---- execution ------------------------------------------------------ */
/* Execute the graph. feed_ids/feed_vals supply values for input ACTIVATION
 * tensor nodes (weights are materialized from storage automatically).
 * grab_ids/grab_vals copy out the final tensor values.
 * Returns 0 on success, -1 on invalid graph / unknown op / NULL buffer. */
int wubu_graph_execute(wubu_graph_t *g,
                       const int    *feed_ids,  const float *const *feed_vals, int n_feed,
                       const int    *grab_ids,  float *const *grab_vals,       int n_grab);

/* ---- op registry (the extensible lattice) --------------------------- */
/* Op function signature: receives the graph (read-only), the op node, and
 * the live activation/weight buffers indexed by tensor node. Ops must not
 * mutate the graph. Register extension kinds with kind >= WUBU_OP_COUNT. */
typedef void (*wubu_graph_op_fn)(const wubu_graph_t *g,
                                 const wubu_graph_op_t *op,
                                 float *const *bufs);

/* Register a handler for `kind`. Returns 0 on success, -1 if the kind is
 * a built-in (those are fixed) or already registered. */
int wubu_graph_register_op(int kind, wubu_graph_op_fn fn);

/* ---- introspection (binders + diagnostics) -------------------------- */
int   wubu_graph_n_tensors(const wubu_graph_t *g);
int   wubu_graph_n_ops(const wubu_graph_t *g);
const wubu_graph_tensor_t *wubu_graph_tensor(const wubu_graph_t *g, int idx);
const wubu_graph_op_t     *wubu_graph_op(const wubu_graph_t *g, int idx);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_GRAPH_H */
