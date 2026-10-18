/*
 * wubu_rlm.h -- bounded recursive decomposition of an input buffer.
 *
 * The RLM result that matters for an OS is structural, not neural: a prompt
 * should not be copied into a fixed window. It should live as an object the
 * controller can slice, recurse over, and merge, with a depth limit and a byte
 * budget. This module is that controller. It does not call a model; the leaf
 * callback is where a model (or a kernel routine) goes.
 *
 * Contract:
 *   - every byte is handed to a leaf at most once;
 *   - recursion stops at max_depth, at chunk-sized nodes, or at the budget;
 *   - a leaf or merge error aborts the whole run (no silent partial answer).
 */
#ifndef WUBU_RLM_H
#define WUBU_RLM_H

#include <stddef.h>

typedef struct {
    size_t chunk;     /* stop splitting when a node is this small */
    int max_depth;    /* 0 = one leaf over the whole input */
    size_t budget;    /* max bytes handed to leaves; 0 = unlimited */
} wubu_rlm_cfg;

typedef struct {
    unsigned long leaves;
    unsigned long nodes;
    unsigned long merges;
    size_t covered;       /* bytes actually processed by leaves */
    int depth_reached;
    int truncated;        /* the budget stopped the walk early */
} wubu_rlm_stats;

/* Process [p, p+n) into out. Return bytes written, or (size_t)-1 on error. */
typedef size_t (*wubu_rlm_leaf_fn)(void *ctx, const char *p, size_t n,
                                   char *out, size_t cap);

/* Combine `count` child results in order. Return bytes written or -1. */
typedef size_t (*wubu_rlm_merge_fn)(void *ctx, const char *const *parts,
                                    const size_t *lens, int count,
                                    char *out, size_t cap);

/* Run the decomposition. Returns 0 on success, -1 on bad arguments or a
 * callback error, -2 if the final result does not fit in `outcap`. */
int wubu_rlm_run(const char *data, size_t len, const wubu_rlm_cfg *cfg,
                 wubu_rlm_leaf_fn leaf, wubu_rlm_merge_fn merge, void *ctx,
                 char *out, size_t outcap, wubu_rlm_stats *st);

#endif /* WUBU_RLM_H */