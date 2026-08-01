/*
 * wubu_coord.h -- Concurrent-agent coordination (AH01-AH04).
 */
#ifndef WUBU_COORD_H
#define WUBU_COORD_H

#define WUBU_LOCK_MAX 64
#define WUBU_LOCK_NAME 128
#define WUBU_LOCK_AGENT 32

typedef struct {
    char file[WUBU_LOCK_MAX][WUBU_LOCK_NAME];
    char owner[WUBU_LOCK_MAX][WUBU_LOCK_AGENT];
    long ts[WUBU_LOCK_MAX];
    int  n;
} wubu_locktab_t;

typedef struct {
    char agents[16][WUBU_LOCK_AGENT];
    int  n;
} wubu_memacl_t;

int  wubu_lock_acquire(wubu_locktab_t *t, const char *file, const char *agent,
                       long now_ms, long stale_ms);
int  wubu_lock_release(wubu_locktab_t *t, const char *file, const char *agent);
int  wubu_txn_committable(const wubu_locktab_t *t, const char **files, int n,
                          const char *agent);
int  wubu_mem_allowed(const wubu_memacl_t *a, const char *agent);
const char *wubu_resolve_conflict(const char *a, int prio_a, const char *b, int prio_b);
int  wubu_heartbeat_alive(long last_ms, long now_ms, long window_ms);

#endif
