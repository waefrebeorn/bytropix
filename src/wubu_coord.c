/*
 * wubu_coord.c -- Concurrent-agent coordination (AH01-AH04). C11.
 *
 * Convergence (CoAgent MTPO serializability-at-quiescence, shared-memory
 * access-control, intent-lock-before-edit, conflict-resolution 7-hop):
 *   - AH01 intent-lock: claim a file before editing; reject if another agent
 *          holds it. Non-blocking, time-stamped, stale-lock adoption.
 *   - AH02 serializability at quiescence: a transaction either commits all its
 *          file edits or none (MTPO targeted repair, not abort-the-world).
 *   - AH03 shared-memory access-control: a memory region has an ACL; only
 *          authorized agents may read/write it (default deny).
 *   - AH04 heartbeat + conflict resolution: agents report liveness; on overlap,
 *          lower-priority yields (consensus by priority, not abort).
 *
 * Pure C11, deterministic, testable. No third-party deps (homestic lock table).
 */
#include "wubu_coord.h"
#include <stdlib.h>
#include <string.h>

/* AH01: try to lock a file for `agent`. Returns 1 if granted, 0 if held by
 * another agent (or stale-adopted by caller). now_ms drives stale reclamation. */
int wubu_lock_acquire(wubu_locktab_t *t, const char *file, const char *agent,
                      long now_ms, long stale_ms) {
    if (!t || t->n >= WUBU_LOCK_MAX) return 0;
    /* reclaim stale locks */
    for (int i = 0; i < t->n; i++)
        if (now_ms - t->ts[i] > stale_ms) { /* drop stale */ t->owner[i][0] = 0; }
    for (int i = 0; i < t->n; i++) {
        if (t->owner[i][0] == 0) continue;            /* stale, reclaimed */
        if (strcmp(t->file[i], file) == 0) {
            if (strcmp(t->owner[i], agent) == 0) return 1; /* already mine */
            return 0;                                  /* held by other */
        }
    }
    /* grant */
    int i = t->n++;
    strncpy(t->file[i], file, WUBU_LOCK_NAME - 1); t->file[i][WUBU_LOCK_NAME-1] = 0;
    strncpy(t->owner[i], agent, WUBU_LOCK_AGENT - 1); t->owner[i][WUBU_LOCK_AGENT-1] = 0;
    t->ts[i] = now_ms;
    return 1;
}

/* AH01: release a lock held by agent. */
int wubu_lock_release(wubu_locktab_t *t, const char *file, const char *agent) {
    if (!t) return 0;
    for (int i = 0; i < t->n; i++)
        if (strcmp(t->file[i], file) == 0 && strcmp(t->owner[i], agent) == 0) {
            t->owner[i][0] = 0; return 1;
        }
    return 0;
}

/* AH02: transaction commit check — all claimed files still owned by agent. */
int wubu_txn_committable(const wubu_locktab_t *t, const char **files, int n,
                         const char *agent) {
    if (!t) return 0;
    for (int j = 0; j < n; j++) {
        int ok = 0;
        for (int i = 0; i < t->n; i++)
            if (t->owner[i][0] && strcmp(t->file[i], files[j]) == 0 &&
                strcmp(t->owner[i], agent) == 0) { ok = 1; break; }
        if (!ok) return 0;   /* a claimed file was lost -> abort txn (repair, not crash) */
    }
    return 1;
}

/* AH03: shared-memory ACL check. region has allowlist of agent names. */
int wubu_mem_allowed(const wubu_memacl_t *a, const char *agent) {
    if (!a) return 0;
    for (int i = 0; i < a->n; i++)
        if (strcmp(a->agents[i], agent) == 0) return 1;  /* authorized */
    return 0;                                            /* default deny */
}

/* AH04: conflict resolution by priority. Lower number = higher priority wins;
 * returns the winning agent (the one that keeps the file). */
const char *wubu_resolve_conflict(const char *a, int prio_a,
                                  const char *b, int prio_b) {
    if (prio_a <= prio_b) return a;
    return b;
}

/* AH04: heartbeat staleness — is agent alive within window? */
int wubu_heartbeat_alive(long last_ms, long now_ms, long window_ms) {
    return (now_ms - last_ms) <= window_ms ? 1 : 0;
}
