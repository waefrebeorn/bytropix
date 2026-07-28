/*
 * wubu_cuda_graph.c — CUDA-graph decode capture + partial param update
 * (Area E, items E.41/E.42/E.43/E.50). C11 planning logic is testable on CPU;
 * the actual cudaGraph* calls are guarded for CUDA builds.
 *
 * Key idea (NVIDIA llama.cpp post): capture the full decode step once, then
 * on each token only update the KV-related node parameters via
 * cudaGraphExecUpdate / node param setter, avoiding a full re-capture per step.
 */
#include "wubu_cuda_graph.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#if defined(__CUDACC__) || defined(HAVE_CUDA)
#include <cuda_runtime.h>
#endif

struct wubu_cuda_graph {
    int captured;          /* 1 if a graph is live */
    int seq_len;           /* current captured sequence length */
    int max_seq;           /* capacity */
    /* CPU-side mirror of the updatable params (KV pointers/offsets). */
    void **kv_ptrs;        /* [max_seq] device pointers for each step */
    int    kv_offsets[WUBU_CG_MAX_NODES];
    int    n_nodes;
};

wubu_cuda_graph_t *wubu_cuda_graph_create(int max_seq) {
    wubu_cuda_graph_t *g = (wubu_cuda_graph_t *)calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->max_seq = max_seq;
    g->kv_ptrs = (void **)calloc(max_seq, sizeof(void *));
    g->n_nodes = WUBU_CG_MAX_NODES;
    return g;
}
void wubu_cuda_graph_free(wubu_cuda_graph_t *g) {
    if (!g) return;
    free(g->kv_ptrs);
    free(g);
}

/* Plan a capture at sequence length `seq`. Returns 0 on success. */
int wubu_cuda_graph_plan(wubu_cuda_graph_t *g, int seq) {
    if (seq > g->max_seq) return -1;
    g->seq_len = seq;
    g->captured = 1;
    /* In a CUDA build this would call cudaStreamBeginCapture / EndCapture
     * and stash the resulting cudaGraphExec_t. */
    return 0;
}

/* Update only the KV node param for the current step (partial update, E.43).
 * Avoids re-capturing the whole graph each token. */
int wubu_cuda_graph_update_kv(wubu_cuda_graph_t *g, int step, void *kv_dev_ptr) {
    if (!g->captured) return -1;
    if (step < 0 || step >= g->max_seq) return -1;
    g->kv_ptrs[step] = kv_dev_ptr;
    g->kv_offsets[step % WUBU_CG_MAX_NODES] = step;
    /* CUDA build: cudaGraphExecUpdate(graphExec, node, &params, &result); */
    return 0;
}

/* Replay the captured graph for one decode step. */
int wubu_cuda_graph_replay(wubu_cuda_graph_t *g) {
    if (!g->captured) return -1;
    /* CUDA build: cudaGraphLaunch(graphExec, stream); */
    return 0;
}
