#ifndef WUBU_SVC_CGROUP_H
#define WUBU_SVC_CGROUP_H

/* wubu_svc_cgroup.h -- KERNEL #15 cgroup v2 per-service via /n/svc.
 *
 * cgroup v2 controllers bound per 9P service: each service registered under
 * /n/svc/<name>/ gets its own cgroup exposing the cpu, memory, and io
 * controllers as writable files (cpu.max, memory.max, io.max). Limits are
 * validated and applied; per-controller usage is accounted and checked against
 * the limit so a caller can detect a throttled / OOM condition. Deterministic,
 * real work (no stubs). */

#include <stddef.h>
#include <stdint.h>

#define WUBU_SVC_MAX 64
#define WUBU_SVC_NAME_MAX 48

typedef struct {
    char name[WUBU_SVC_NAME_MAX];
    int present;
    /* cpu controller: quota/period in microseconds; 0 period = unlimited */
    int64_t cpu_quota_us;
    int64_t cpu_period_us;
    uint64_t cpu_used_us;     /* accounted cpu time */
    /* memory controller: bytes; 0 = unlimited */
    uint64_t mem_max;
    uint64_t mem_used;
    /* io controller: bytes/sec; 0 = unlimited */
    uint64_t io_max_bps;
    uint64_t io_used_bytes;
} wubu_svc_cgroup_t;

typedef struct {
    wubu_svc_cgroup_t svcs[WUBU_SVC_MAX];
    int n;
} wubu_svc_cgroups_t;

/* Initialise. Returns 0. */
int wubu_svc_init(wubu_svc_cgroups_t *s);

/* Register a service cgroup (all limits start unlimited). Returns 0/-1. */
int wubu_svc_add(wubu_svc_cgroups_t *s, const char *name);

/* Find a service index, or -1. */
int wubu_svc_find(const wubu_svc_cgroups_t *s, const char *name);

/* Set cpu.max (quota/period, microseconds). Period must be > 0; quota >= 0
 * (0 = no cpu). Returns 0/-1 (fail-closed on bad args). */
int wubu_svc_set_cpu(wubu_svc_cgroups_t *s, const char *name,
                     int64_t quota_us, int64_t period_us);

/* Set memory.max (bytes; 0 = unlimited). Returns 0/-1. */
int wubu_svc_set_mem(wubu_svc_cgroups_t *s, const char *name, uint64_t max_bytes);

/* Set io.max (bytes/sec; 0 = unlimited). Returns 0/-1. */
int wubu_svc_set_io(wubu_svc_cgroups_t *s, const char *name, uint64_t max_bps);

/* Account usage. Returns 0/-1. cpu_us/mem_bytes/io_bytes are deltas. */
int wubu_svc_account(wubu_svc_cgroups_t *s, const char *name,
                     uint64_t cpu_us, uint64_t mem_bytes, uint64_t io_bytes);

/* Is the service over its cpu quota (used >= quota, quota>0)? 1/0/-1. */
int wubu_svc_cpu_throttled(const wubu_svc_cgroups_t *s, const char *name);

/* Is the service over its memory limit (used > max, max>0)? 1/0/-1. */
int wubu_svc_mem_exceeded(const wubu_svc_cgroups_t *s, const char *name);

/* Is the service over its io limit (used > max, max>0)? 1/0/-1. */
int wubu_svc_io_exceeded(const wubu_svc_cgroups_t *s, const char *name);

#endif
