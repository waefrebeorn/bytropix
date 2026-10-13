#ifndef WUBU_CT_SANDBOX_H
#define WUBU_CT_SANDBOX_H

/* wubu_ct_sandbox.h -- RUNTIME #341 container isolation (cgroups v2 + seccomp).
 *
 * A complete container sandbox: one spec binds cgroup-v2 resource controllers
 * (cpu, memory, pids, io) AND a seccomp syscall filter. A syscall is admitted
 * only if it passes the seccomp allowlist; resource usage is accounted against
 * the cgroup limits so the sandbox reports when a container is cpu-throttled,
 * memory-exceeded, pids-exceeded, or io-exceeded. This composes the per-service
 * cgroup model (#15) with a default-deny syscall filter (the seccomp layer).
 * Deterministic, real work (no stubs). */

#include <stddef.h>
#include <stdint.h>

#define WUBU_SB_MAX_CONTAINERS 32
#define WUBU_SB_MAX_SYSCALLS 256
#define WUBU_SB_NAME_MAX 48

typedef struct {
    char name[WUBU_SB_NAME_MAX];
    int present;
    /* cgroup v2 controllers */
    int64_t cpu_quota_us;   int64_t cpu_period_us;   /* 0 period = unlimited */
    uint64_t mem_max;       /* 0 = unlimited */
    uint32_t pids_max;      /* 0 = unlimited */
    uint64_t io_max_bps;    /* 0 = unlimited */
    /* usage */
    uint64_t cpu_used_us;
    uint64_t mem_used;
    uint32_t pids_used;
    uint64_t io_used_bytes;
    /* seccomp allowlist */
    int allow[WUBU_SB_MAX_SYSCALLS];
    int n_allow;
    int seccomp_default_deny;  /* 1 = unlisted syscalls denied */
    /* stats */
    uint64_t n_admitted;
    uint64_t n_blocked;
} wubu_sb_ct_t;

typedef struct {
    wubu_sb_ct_t cts[WUBU_SB_MAX_CONTAINERS];
    int n;
} wubu_sandbox_t;

/* Initialise. Returns 0. */
int wubu_sb_init(wubu_sandbox_t *s);

/* Create a container with cgroup limits (0 = unlimited for each) and a
 * default-deny seccomp policy. Returns 0/-1. */
int wubu_sb_create(wubu_sandbox_t *s, const char *name,
                   int64_t cpu_quota_us, int64_t cpu_period_us,
                   uint64_t mem_max, uint32_t pids_max, uint64_t io_max_bps);

/* Find a container index, or -1. */
int wubu_sb_find(const wubu_sandbox_t *s, const char *name);

/* Allow a syscall number in the seccomp filter. Returns 0/-1. */
int wubu_sb_allow_syscall(wubu_sandbox_t *s, const char *name, int syscall_nr);

/* Set seccomp default (1 = deny unlisted, 0 = allow unlisted). Returns 0/-1. */
int wubu_sb_set_default_deny(wubu_sandbox_t *s, const char *name, int deny);

/* Try to perform `syscall_nr` in the container. Returns 1 if admitted (passes
 * seccomp), 0 if blocked by seccomp, -1 on bad args. Admission is independent
 * of resource limits (limits are reported separately, not enforced here). */
int wubu_sb_syscall(wubu_sandbox_t *s, const char *name, int syscall_nr);

/* Account resource usage (deltas). Returns 0/-1. */
int wubu_sb_account(wubu_sandbox_t *s, const char *name,
                    uint64_t cpu_us, uint64_t mem_bytes,
                    uint32_t pids, uint64_t io_bytes);

/* Resource-limit checks (1 = exceeded/throttled, 0 = ok, -1 = bad args). */
int wubu_sb_cpu_throttled(const wubu_sandbox_t *s, const char *name);
int wubu_sb_mem_exceeded(const wubu_sandbox_t *s, const char *name);
int wubu_sb_pids_exceeded(const wubu_sandbox_t *s, const char *name);
int wubu_sb_io_exceeded(const wubu_sandbox_t *s, const char *name);

/* Overall isolation verdict: 1 if the container is within all limits AND has a
 * seccomp filter active, 0 if any limit exceeded, -1 bad args. */
int wubu_sb_isolated(const wubu_sandbox_t *s, const char *name);

#endif
