/*
 * test_rlm.c -- bounded recursive decomposition (RLM controller).
 *
 * The invariants that matter: every byte reaches exactly one leaf, recursion
 * stops at the depth limit, the budget is honoured exactly, merge order is
 * preserved, and a leaf error aborts rather than returning a partial answer.
 */
#include "wubu_rlm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else printf("  [PASS] %s\n", m); } while (0)

typedef struct {
    unsigned long seq;
    size_t sum;
    int poison;
} ctx_t;

static size_t leaf_cb(void *v, const char *p, size_t n, char *out, size_t cap)
{
    ctx_t *c = (ctx_t *)v;
    if (c->poison && p[0] == 'X') return (size_t)-1;
    c->sum += n;
    int k = snprintf(out, cap, "%lu", c->seq++);
    if (k < 0 || (size_t)k >= cap) return (size_t)-1;
    return (size_t)k;
}

static size_t merge_cb(void *v, const char *const *parts, const size_t *lens,
                       int count, char *out, size_t cap)
{
    (void)v;
    size_t w = 0;
    for (int i = 0; i < count; i++) {
        if (w + lens[i] + 2 > cap) return (size_t)-1;
        memcpy(out + w, parts[i], lens[i]);
        w += lens[i];
        if (i + 1 < count) out[w++] = ',';
    }
    out[w] = '\0';
    return w;
}

int main(void)
{
    const size_t N = 1u << 20;
    char *data = (char *)malloc(N);
    if (!data) { printf("alloc failed\n"); return 1; }
    memset(data, 'A', N);

    char out[4096];
    wubu_rlm_stats st;
    wubu_rlm_cfg cfg;
    ctx_t ctx;

    printf("=== test_rlm (recursive decomposition) ===\n");

    /* 1. depth 0: one leaf over the whole input. */
    memset(&ctx, 0, sizeof ctx);
    cfg.chunk = 4096; cfg.max_depth = 0; cfg.budget = 0;
    CHECK(wubu_rlm_run(data, N, &cfg, leaf_cb, merge_cb, &ctx, out, sizeof out, &st) == 0,
          "depth 0 runs");
    CHECK(st.leaves == 1 && st.merges == 0 && st.depth_reached == 0,
          "depth 0 -> 1 leaf, 0 merges");
    CHECK(st.covered == N && ctx.sum == N, "depth 0 covers every byte once");

    /* 2. depth 8: 2^8 leaves, 255 merges, all bytes. */
    memset(&ctx, 0, sizeof ctx);
    cfg.max_depth = 8;
    CHECK(wubu_rlm_run(data, N, &cfg, leaf_cb, merge_cb, &ctx, out, sizeof out, &st) == 0,
          "depth 8 runs");
    CHECK(st.leaves == 256, "depth 8 -> 256 leaves");
    CHECK(st.merges == 255, "depth 8 -> 255 merges");
    CHECK(st.depth_reached == 8, "depth reached 8");
    CHECK(st.covered == N && ctx.sum == N, "every byte covered exactly once");
    CHECK(strncmp(out, "0,1,2", 5) == 0, "merge preserves left-to-right order");

    /* 3. budget: stops exactly at the limit and says so. */
    memset(&ctx, 0, sizeof ctx);
    cfg.budget = 100000;
    CHECK(wubu_rlm_run(data, N, &cfg, leaf_cb, merge_cb, &ctx, out, sizeof out, &st) == 0,
          "budgeted run completes");
    CHECK(st.truncated == 1, "truncated flag set");
    CHECK(st.covered == 100000 && ctx.sum == 100000, "budget honoured exactly");
    CHECK(st.leaves == 25, "24 full leaves + one partial");

    /* 4. leaf error aborts the run. */
    memset(&ctx, 0, sizeof ctx);
    ctx.poison = 1;
    data[0] = 'X';
    cfg.budget = 0;
    CHECK(wubu_rlm_run(data, N, &cfg, leaf_cb, merge_cb, &ctx, out, sizeof out, &st) == -1,
          "leaf error -> -1, no partial answer");
    data[0] = 'A';

    /* 5. output too small is refused, not truncated silently. */
    memset(&ctx, 0, sizeof ctx);
    char tiny[2];
    CHECK(wubu_rlm_run(data, N, &cfg, leaf_cb, merge_cb, &ctx, tiny, sizeof tiny, &st) == -2,
          "output overflow -> -2");

    /* 6. chunk rule: one leaf when the input is smaller than the chunk. */
    memset(&ctx, 0, sizeof ctx);
    cfg.chunk = N; cfg.max_depth = 8;
    CHECK(wubu_rlm_run(data, N, &cfg, leaf_cb, merge_cb, &ctx, out, sizeof out, &st) == 0 &&
          st.leaves == 1, "chunk >= input stops at one leaf");

    /* 7. argument validation. */
    CHECK(wubu_rlm_run(NULL, N, &cfg, leaf_cb, merge_cb, &ctx, out, sizeof out, &st) == -1,
          "NULL data -> -1");
    cfg.chunk = 0;
    CHECK(wubu_rlm_run(data, N, &cfg, leaf_cb, merge_cb, &ctx, out, sizeof out, &st) == -1,
          "zero chunk -> -1");

    free(data);
    printf("\n=== test_rlm: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}