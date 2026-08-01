/*
 * wubu_db_cross.h -- Cross-discipline DB -> KV engine mappings (O14/O18/O19).
 */
#ifndef WUBU_DB_CROSS_H
#define WUBU_DB_CROSS_H

#include <stddef.h>

#define WUBU_PLAN_MAX 64

typedef struct {
    int type;     /* caller-defined op tag (e.g. 0=prefill,1=decode,2=evict) */
    double cost;  /* estimated cost; planner orders ascending */
} wubu_op_t;

/* O14 cost-based decode scheduler: returns ops sorted by ascending cost. */
int wubu_plan_decode(const wubu_op_t *ops, int n, int *out_order);

/* O18 provable OOM-never invariant: 1 iff KV@(max_ctx,batch) <= ram. */
int wubu_kv_invariant_ok(int L, int n_kv, int d_h, int b_kv,
                         int batch, int max_ctx, double ram);

/* O19 WAL append-log for KV ops. */
typedef struct {
    int block_id;
    int len;
    long seq;
} wubu_wal_op_t;

typedef struct wubu_wal wubu_wal_t;
wubu_wal_t *wubu_wal_create(int cap);
int wubu_wal_append(wubu_wal_t *w, int block_id, int len, long seq);
int wubu_wal_mark_applied(wubu_wal_t *w);
int wubu_wal_replay(const wubu_wal_t *w, wubu_wal_op_t *out, int out_cap);
void wubu_wal_destroy(wubu_wal_t *w);

#endif /* WUBU_DB_CROSS_H */
