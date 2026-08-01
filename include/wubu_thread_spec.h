/*
 * wubu_thread_spec.h — CPU thread-specialization analog (doc H02).
 *
 * GPU "warp/thread specialization" splits work so different thread groups
 * handle different phases (e.g. one group does prefill GEMM, another does
 * decode GEMV) without context-switch thrash. On CPU we approximate this with
 * two pinned thread pools:
 *   - PREFILL pool: large batched matmul work (compute-bound)
 *   - DECODE  pool: small GEMV / attention work (latency-bound)
 * Each pool is pinned to a disjoint set of cores (via pthread affinity) so the
 * OS scheduler never migrates a prefill thread onto a decode core. This removes
 * cache-cold migrations and keeps decode tail latency bounded (doc 007 win).
 *
 * Self-contained C11 + POSIX threads. No third-party deps.
 */
#ifndef WUBU_THREAD_SPEC_H
#define WUBU_THREAD_SPEC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WUBU_TS_PREFILL = 0,
    WUBU_TS_DECODE  = 1
} wubu_ts_role_t;

typedef void (*wubu_ts_job_fn)(void *arg);

typedef struct wubu_thread_spec wubu_thread_spec_t;

/* Create a specialization controller.
 * prefill_cores / decode_cores: comma-or-space separated core lists, e.g.
 *   "0-3"  -> cores 0,1,2,3
 *   "4,5,6,7" -> cores 4,5,6,7
 * Passing NULL lets the OS schedule freely (still uses two pools, no pinning). */
wubu_thread_spec_t *wubu_thread_spec_create(const char *prefill_cores,
                                            const char *decode_cores);

void wubu_thread_spec_free(wubu_thread_spec_t *ts);

/* Submit a job to a role's pool. Returns 0 on accept, -1 on failure.
 * The call is asynchronous; use wubu_thread_spec_wait(role) to block until
 * all outstanding jobs for that role complete. */
int wubu_thread_spec_submit(wubu_thread_spec_t *ts, wubu_ts_role_t role,
                            wubu_ts_job_fn fn, void *arg);

/* Block until all jobs submitted to `role` have finished. */
void wubu_thread_spec_wait(wubu_thread_spec_t *ts, wubu_ts_role_t role);

/* Stats: number of cores assigned to each role. */
void wubu_thread_spec_cores(const wubu_thread_spec_t *ts,
                            int *prefill_n, int *decode_n);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_THREAD_SPEC_H */
