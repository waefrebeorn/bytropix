/*
 * wubu_rlm.c -- bounded recursive decomposition. See wubu_rlm.h.
 */
#include "wubu_rlm.h"
#include <stdlib.h>
#include <string.h>

#define NODE_CAP 4096

typedef struct {
    const wubu_rlm_cfg *cfg;
    wubu_rlm_leaf_fn leaf;
    wubu_rlm_merge_fn merge;
    void *ctx;
    wubu_rlm_stats *st;
} env_t;

static char *empty_result(size_t *len)
{
    char *z = (char *)malloc(1);
    if (z) z[0] = '\0';
    *len = 0;
    return z;
}

static char *run_node(env_t *e, const char *p, size_t n, int depth, size_t *len)
{
    e->st->nodes++;
    if (depth > e->st->depth_reached) e->st->depth_reached = depth;
    if (n == 0) return empty_result(len);

    int is_leaf = (n <= e->cfg->chunk) || (depth >= e->cfg->max_depth);
    if (is_leaf) {
        size_t take = n;
        if (e->cfg->budget) {
            if (e->st->covered >= e->cfg->budget) {
                e->st->truncated = 1;
                return empty_result(len);
            }
            size_t left = e->cfg->budget - e->st->covered;
            if (take > left) { take = left; e->st->truncated = 1; }
        }
        if (take == 0) return empty_result(len);

        char *buf = (char *)malloc(NODE_CAP);
        if (!buf) return NULL;
        size_t w = e->leaf(e->ctx, p, take, buf, NODE_CAP);
        if (w == (size_t)-1) { free(buf); return NULL; }
        e->st->leaves++;
        e->st->covered += take;
        *len = w;
        return buf;
    }

    size_t mid = n / 2;
    size_t llen = 0, rlen = 0;
    char *l = run_node(e, p, mid, depth + 1, &llen);
    if (!l) return NULL;
    char *r = run_node(e, p + mid, n - mid, depth + 1, &rlen);
    if (!r) { free(l); return NULL; }

    const char *parts[2] = { l, r };
    size_t lens[2] = { llen, rlen };
    char *buf = (char *)malloc(NODE_CAP);
    if (!buf) { free(l); free(r); return NULL; }
    size_t w = e->merge(e->ctx, parts, lens, 2, buf, NODE_CAP);
    free(l); free(r);
    if (w == (size_t)-1) { free(buf); return NULL; }

    e->st->merges++;
    *len = w;
    return buf;
}

int wubu_rlm_run(const char *data, size_t len, const wubu_rlm_cfg *cfg,
                 wubu_rlm_leaf_fn leaf, wubu_rlm_merge_fn merge, void *ctx,
                 char *out, size_t outcap, wubu_rlm_stats *st)
{
    if (!data || !cfg || !leaf || !merge || !st) return -1;
    if (cfg->chunk == 0 || cfg->max_depth < 0) return -1;
    if (!out && outcap) return -1;

    memset(st, 0, sizeof *st);
    env_t e = { cfg, leaf, merge, ctx, st };

    size_t rlen = 0;
    char *res = run_node(&e, data, len, 0, &rlen);
    if (!res) return -1;
    if (rlen > outcap) { free(res); return -2; }
    if (rlen) memcpy(out, res, rlen);
    if (out && outcap) out[rlen < outcap ? rlen : outcap - 1] = '\0';
    free(res);
    return 0;
}