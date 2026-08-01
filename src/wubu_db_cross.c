/*
 * wubu_db_cross.c -- Cross-discipline DB -> KV engine mappings (O14 / O18 / O19).
 * C11, no third-party deps.
 *
 * Convergence (DB query-plan + WAL + formal-verification 7-hop): these three
 * gaps each import a database concept into the KV/decode engine as a small,
 * testable primitive:
 *   - O14 DB query-plan -> decode schedule: a cost-based optimizer that, given a
 *        set of pending operations (prefill / decode / evict) with costs, returns
 *        them in lowest-cost-first order (like a query planner picking a plan).
 *   - O18 Formal OOM-never invariant: a *verifiable* guard -- given model params,
 *        max context, batch and RAM, prove KV_bytes(max_ctx,batch) <= ram. This is
 *        the "provable OOM-never" bound (ties 512k objective + capacity_wall).
 *   - O19 DB WAL -> KV append-log replay: a write-ahead log of KV append ops that
 *        can be replayed to reconstruct state after a crash (durability primitive).
 *
 * Triple-DA: null/zero handled; no div-by-zero; WAL replay is deterministic and
 * bounds-checked (no OOB even on a truncated/corrupt log).
 */
#include "wubu_db_cross.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct wubu_wal {
    wubu_wal_op_t *ops;
    int cap;
    int n;
    int applied;
};

/* O14 cost-based decode-schedule planner.
 * ops[i] = {type, cost}. Sorts (insertion sort, n small) by ascending cost.
 * type is just a tag we forward to the caller's executor. Writes the reordered
 * indices into out_order (caller-sized >= n). Returns n. */
int wubu_plan_decode(const wubu_op_t *ops, int n, int *out_order) {
    if (!ops || !out_order || n <= 0) return 0;
    if (n > WUBU_PLAN_MAX) n = WUBU_PLAN_MAX;
    /* init order identity */
    for (int i = 0; i < n; i++) out_order[i] = i;
    /* insertion sort by cost ascending */
    for (int i = 1; i < n; i++) {
        int key = out_order[i];
        int j = i - 1;
        while (j >= 0 && ops[out_order[j]].cost > ops[key].cost) {
            out_order[j + 1] = out_order[j];
            j--;
        }
        out_order[j + 1] = key;
    }
    return n;
}

/* O18 provable OOM-never invariant.
 * Returns 1 iff KV bytes at (max_ctx, batch) with given model geometry is <= ram.
 * This is the formal guard: if it holds, the engine can never OOM on KV at this
 * context/batch (assuming it never allocates more than the computed budget). */
int wubu_kv_invariant_ok(int L, int n_kv, int d_h, int b_kv,
                         int batch, int max_ctx, double ram) {
    if (L <= 0 || n_kv <= 0 || d_h <= 0 || b_kv < 0 || batch <= 0 ||
        max_ctx <= 0 || ram <= 0.0)
        return 0;
    double per_tok_layer = 2.0 * n_kv * d_h * (b_kv / 8.0);
    double kv = per_tok_layer * L * batch * max_ctx;
    return (kv <= ram) ? 1 : 0;
}

/* O19 WAL append-log for KV ops. */
wubu_wal_t *wubu_wal_create(int cap) {
    if (cap <= 0) cap = 64;
    wubu_wal_t *w = (wubu_wal_t *)calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->ops = (wubu_wal_op_t *)calloc((size_t)cap, sizeof(wubu_wal_op_t));
    if (!w->ops) { free(w); return NULL; }
    w->cap = cap; w->n = 0; w->applied = 0;
    return w;
}

/* Append an op to the log (not yet applied). Returns 1 on success. */
int wubu_wal_append(wubu_wal_t *w, int block_id, int len, long seq) {
    if (!w || len < 0) return 0;
    if (w->n >= w->cap) return 0;            /* full: caller must flush */
    w->ops[w->n].block_id = block_id;
    w->ops[w->n].len = len;
    w->ops[w->n].seq = seq;
    w->n++;
    return 1;
}

/* Mark ops [applied, n) as applied (durable). Returns count applied. */
int wubu_wal_mark_applied(wubu_wal_t *w) {
    if (!w) return 0;
    int c = w->n - w->applied;
    w->applied = w->n;
    return c;
}

/* Replay un-applied ops into out (caller-sized >= cap). Returns count replayed.
 * Bounds-checked: never reads beyond n. Stable order. */
int wubu_wal_replay(const wubu_wal_t *w, wubu_wal_op_t *out, int out_cap) {
    if (!w || !out || out_cap <= 0) return 0;
    int cnt = 0;
    for (int i = w->applied; i < w->n && cnt < out_cap; i++)
        out[cnt++] = w->ops[i];
    return cnt;
}

void wubu_wal_destroy(wubu_wal_t *w) {
    if (!w) return;
    free(w->ops);
    free(w);
}
