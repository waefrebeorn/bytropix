// src/wubu_graph.c
// Universal graph IR implementation — the mathematical-engine subtext.
//
// A model is a directed graph of typed tensor operations. Tensor nodes
// carry wubu_weight_t descriptors (storage-backed for weights, executor-
// owned buffers for activations); op nodes dispatch through a registry;
// the executor walks ops in topological order.
//
// The universal-manifold contract (docs/universal-manifold-design.md):
//   * weights are materialized through ONE materializer
//     (wubu_weight_to_f32) and consumed through ONE matmul
//     (wubu_weight_matmul) — no per-format branch anywhere;
//   * new modalities (music/video/diffusion) add ops to the registry
//     and binders that emit graphs — never engine branches.

#include "wubu_graph.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- internal graph layout ------------------------------------------ */
struct wubu_graph {
    wubu_graph_tensor_t *tensors;
    int   n_tensors, cap_tensors;
    wubu_graph_op_t     *ops;
    int   n_ops, cap_ops;
};

/* ---- op registry ---------------------------------------------------- */
/* Built-ins are in a static table; extension kinds (>= WUBU_OP_COUNT)
 * register at runtime. Registry size is generous but bounded. */
#define WUBU_GRAPH_REG_MAX 256

static wubu_graph_op_fn g_op_fns[WUBU_GRAPH_REG_MAX];

static void op_matmul(const wubu_graph_t *g, const wubu_graph_op_t *op, float *const *bufs);
static void op_relu  (const wubu_graph_t *g, const wubu_graph_op_t *op, float *const *bufs);
static void op_add   (const wubu_graph_t *g, const wubu_graph_op_t *op, float *const *bufs);
static void op_silu  (const wubu_graph_t *g, const wubu_graph_op_t *op, float *const *bufs);

static void init_builtin_ops(void) {
    static int done = 0;
    if (done) return;
    g_op_fns[WUBU_OP_MATMUL] = op_matmul;
    g_op_fns[WUBU_OP_RELU]   = op_relu;
    g_op_fns[WUBU_OP_ADD]    = op_add;
    g_op_fns[WUBU_OP_SILU]   = op_silu;
    done = 1;
}

/* ---- helpers -------------------------------------------------------- */
static void free_bufs(float **bufs, int n);

static int64_t tensor_elems(const wubu_graph_tensor_t *t) {
    int64_t n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= t->dims[d];
    return n;
}

/* ---- construction --------------------------------------------------- */
wubu_graph_t *wubu_graph_create(void) {
    init_builtin_ops();
    wubu_graph_t *g = (wubu_graph_t *)calloc(1, sizeof(*g));
    return g;
}

void wubu_graph_free(wubu_graph_t *g) {
    if (!g) return;
    free(g->tensors);
    free(g->ops);
    free(g);
}

static wubu_graph_tensor_t *grow_tensor(wubu_graph_t *g) {
    if (g->n_tensors == g->cap_tensors) {
        int nc = g->cap_tensors ? g->cap_tensors * 2 : 16;
        wubu_graph_tensor_t *nt = (wubu_graph_tensor_t *)
            realloc(g->tensors, (size_t)nc * sizeof(*nt));
        if (!nt) return NULL;
        g->tensors = nt;
        g->cap_tensors = nc;
    }
    return &g->tensors[g->n_tensors];
}

static wubu_graph_op_t *grow_op(wubu_graph_t *g) {
    if (g->n_ops == g->cap_ops) {
        int nc = g->cap_ops ? g->cap_ops * 2 : 16;
        wubu_graph_op_t *no = (wubu_graph_op_t *)
            realloc(g->ops, (size_t)nc * sizeof(*no));
        if (!no) return NULL;
        g->ops = no;
        g->cap_ops = nc;
    }
    return &g->ops[g->n_ops];
}

int wubu_graph_add_tensor(wubu_graph_t *g, const char *name,
                          const wubu_weight_t *w,
                          const int64_t *dims, int n_dims) {
    if (!g || !dims || n_dims < 1 || n_dims > 4) return -1;
    wubu_graph_tensor_t *t = grow_tensor(g);
    if (!t) return -1;
    memset(t, 0, sizeof(*t));
    if (name) {
        strncpy(t->name, name, sizeof(t->name) - 1);
        t->name[sizeof(t->name) - 1] = '\0';
    }
    t->n_dims = n_dims;
    for (int d = 0; d < n_dims; d++) t->dims[d] = dims[d];
    if (w && w->data) {
        t->w = *w;
        t->is_weight = 1;
    }
    return g->n_tensors++;
}

int wubu_graph_add_op(wubu_graph_t *g, int kind,
                      const int *inputs, int n_inputs, int output) {
    if (!g || !inputs || n_inputs < 1 || n_inputs > 8) return -1;
    if (output < 0 || output >= g->n_tensors) return -1;
    for (int i = 0; i < n_inputs; i++) {
        if (inputs[i] < 0 || inputs[i] >= g->n_tensors) return -1;
        if (inputs[i] == output) return -1; /* no in-place ops in the IR */
    }
    if (kind >= 0 && kind < WUBU_OP_COUNT && !g_op_fns[kind]) return -1;
    wubu_graph_op_t *op = grow_op(g);
    if (!op) return -1;
    memset(op, 0, sizeof(*op));
    op->kind = kind;
    op->n_inputs = n_inputs;
    for (int i = 0; i < n_inputs; i++) op->inputs[i] = inputs[i];
    op->output = output;
    return g->n_ops++;
}

int wubu_graph_validate(const wubu_graph_t *g, int *bad_op) {
    if (!g) return -1;
    int no = g->n_ops, nt = g->n_tensors;
    if (no == 0) return 0;

    /* This IR executes ops strictly in index order, so the producer of
     * a tensor must run BEFORE every consumer. min_writer[ti] = the
     * earliest op that writes tensor ti; -1 means nobody writes it
     * (a caller-supplied feed). */
    int *min_writer = (int *)malloc((size_t)(nt ? nt : 1) * sizeof(int));
    if (!min_writer) return -1;
    for (int i = 0; i < nt; i++) min_writer[i] = -1;
    for (int o = 0; o < no; o++) {
        int out = g->ops[o].output;
        if (min_writer[out] < 0 || o < min_writer[out]) min_writer[out] = o;
    }

    /* Every non-weight input must be either a feed (never written) or
     * produced by an op that runs strictly earlier. A writer at or
     * after this op is a forward reference = cycle / order violation. */
    int rc = 0;
    for (int o = 0; o < no; o++) {
        const wubu_graph_op_t *op = &g->ops[o];
        for (int i = 0; i < op->n_inputs; i++) {
            int ti = op->inputs[i];
            const wubu_graph_tensor_t *t = &g->tensors[ti];
            if (t->is_weight) continue;
            if (min_writer[ti] >= 0 && min_writer[ti] >= o) {
                if (bad_op) *bad_op = o;
                rc = -1;
                goto out;
            }
        }
    }
out:
    free(min_writer);
    return rc;
}

/* ---- execution ------------------------------------------------------ */
/* Allocate one F32 buffer per tensor. Weights materialize via the
 * universal materializer; activations start zeroed, then feeds and ops
 * write them. */
static float **alloc_buffers(const wubu_graph_t *g) {
    float **bufs = (float **)calloc((size_t)(g->n_tensors ? g->n_tensors : 1), sizeof(float *));
    if (!bufs) return NULL;
    for (int i = 0; i < g->n_tensors; i++) {
        const wubu_graph_tensor_t *t = &g->tensors[i];
        int64_t n = tensor_elems(t);
        if (n <= 0) continue;
        bufs[i] = (float *)calloc((size_t)n, sizeof(float));
        if (!bufs[i]) { free_bufs(bufs, g->n_tensors); return NULL; }
        if (t->is_weight) {
            if (wubu_weight_to_f32(&t->w, bufs[i]) != 0) {
                /* unknown type: leave zeroed and let the audit flag it */
                continue;
            }
        }
    }
    return bufs;
}

int wubu_graph_execute(wubu_graph_t *g,
                       const int    *feed_ids,  const float *const *feed_vals, int n_feed,
                       const int    *grab_ids,  float *const *grab_vals,       int n_grab) {
    if (!g) return -1;
    int bad = -1;
    if (wubu_graph_validate(g, &bad) != 0) return -1;
    for (int i = 0; i < n_feed; i++)
        if (!feed_vals[i] || feed_ids[i] < 0 || feed_ids[i] >= g->n_tensors) return -1;
    for (int i = 0; i < n_grab; i++)
        if (!grab_vals[i] || grab_ids[i] < 0 || grab_ids[i] >= g->n_tensors) return -1;

    float **bufs = alloc_buffers(g);
    if (!bufs) return -1;

    /* feeds overwrite activation placeholders */
    for (int i = 0; i < n_feed; i++) {
        const wubu_graph_tensor_t *t = &g->tensors[feed_ids[i]];
        int64_t n = tensor_elems(t);
        if (n > 0) memcpy(bufs[feed_ids[i]], feed_vals[i], (size_t)n * sizeof(float));
    }

    /* walk ops in topological order */
    for (int o = 0; o < g->n_ops; o++) {
        const wubu_graph_op_t *op = &g->ops[o];
        wubu_graph_op_fn fn = (op->kind >= 0 && op->kind < WUBU_GRAPH_REG_MAX)
                                ? g_op_fns[op->kind] : NULL;
        if (!fn) { free_bufs(bufs, g->n_tensors); return -1; }
        fn(g, op, bufs);
    }

    /* copy requested outputs */
    for (int i = 0; i < n_grab; i++) {
        const wubu_graph_tensor_t *t = &g->tensors[grab_ids[i]];
        int64_t n = tensor_elems(t);
        if (n > 0) memcpy(grab_vals[i], bufs[grab_ids[i]], (size_t)n * sizeof(float));
    }

    free_bufs(bufs, g->n_tensors);
    return 0;
}

static void free_bufs(float **bufs, int n) {
    if (!bufs) return;
    for (int i = 0; i < n; i++) free(bufs[i]);
    free(bufs);
}

int wubu_graph_register_op(int kind, wubu_graph_op_fn fn) {
    init_builtin_ops();
    if (kind < 0 || kind >= WUBU_GRAPH_REG_MAX) return -1;
    if (kind < WUBU_OP_COUNT) return -1;             /* built-ins are fixed */
    if (g_op_fns[kind]) return -1;                   /* already registered   */
    g_op_fns[kind] = fn;
    return 0;
}

/* ---- introspection --------------------------------------------------- */
int wubu_graph_n_tensors(const wubu_graph_t *g) { return g ? g->n_tensors : 0; }
int wubu_graph_n_ops(const wubu_graph_t *g)     { return g ? g->n_ops : 0; }

const wubu_graph_tensor_t *wubu_graph_tensor(const wubu_graph_t *g, int idx) {
    if (!g || idx < 0 || idx >= g->n_tensors) return NULL;
    return &g->tensors[idx];
}

const wubu_graph_op_t *wubu_graph_op(const wubu_graph_t *g, int idx) {
    if (!g || idx < 0 || idx >= g->n_ops) return NULL;
    return &g->ops[idx];
}

/* ---- built-in ops ---------------------------------------------------- */
static void op_matmul(const wubu_graph_t *g, const wubu_graph_op_t *op, float *const *bufs) {
    /* y[n] = x[k] @ W[n,k]. n (out) and k (in) come from the SECOND
     * operand's dims: dims[0]=n, dims[1]=k — the same rule whether W is
     * a quantized weight (universal matmul) or an activation (SGEMM). */
    const wubu_graph_tensor_t *tx = &g->tensors[op->inputs[0]];
    const wubu_graph_tensor_t *tw = &g->tensors[op->inputs[1]];
    const wubu_graph_tensor_t *ty = &g->tensors[op->output];
    int64_t n = tw->n_dims > 0 ? tw->dims[0] : 0;   /* out */
    int64_t k = tw->n_dims > 1 ? tw->dims[1] : 0;   /* in  */
    if (k <= 0 || n <= 0) return;
    if (tensor_elems(tx) < k || tensor_elems(ty) < n) return;  /* shape guard */
    if (tw->is_weight) {
        wubu_weight_matmul(bufs[op->inputs[0]], &tw->w, k, n, bufs[op->output]);
    } else {
        const float *x = bufs[op->inputs[0]];
        const float *W = bufs[op->inputs[1]];
        float *y = bufs[op->output];
        for (int64_t j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int64_t i = 0; i < k; i++) acc += x[i] * W[j * k + i];
            y[j] = acc;
        }
    }
}

static void op_relu(const wubu_graph_t *g, const wubu_graph_op_t *op, float *const *bufs) {
    const wubu_graph_tensor_t *tx = &g->tensors[op->inputs[0]];
    int64_t n = tensor_elems(tx);
    const float *x = bufs[op->inputs[0]];
    float *y = bufs[op->output];
    for (int64_t i = 0; i < n; i++) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

static void op_add(const wubu_graph_t *g, const wubu_graph_op_t *op, float *const *bufs) {
    const wubu_graph_tensor_t *ta = &g->tensors[op->inputs[0]];
    int64_t n = tensor_elems(ta);
    const float *a = bufs[op->inputs[0]];
    const float *b = bufs[op->inputs[1]];
    float *y = bufs[op->output];
    for (int64_t i = 0; i < n; i++) y[i] = a[i] + b[i];
}

static void op_silu(const wubu_graph_t *g, const wubu_graph_op_t *op, float *const *bufs) {
    const wubu_graph_tensor_t *tx = &g->tensors[op->inputs[0]];
    int64_t n = tensor_elems(tx);
    const float *x = bufs[op->inputs[0]];
    float *y = bufs[op->output];
    for (int64_t i = 0; i < n; i++) {
        float s = 1.0f / (1.0f + expf(-x[i]));
        y[i] = x[i] * s;
    }
}
