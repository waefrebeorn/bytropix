/* wubu_ct_sandbox.c -- RUNTIME #341 container isolation (cgroups v2 + seccomp).
 * See wubu_ct_sandbox.h. Real, deterministic sandbox policy engine. */

#include "wubu_ct_sandbox.h"
#include <string.h>

int wubu_sb_init(wubu_sandbox_t *s)
{
    if (!s) return -1;
    memset(s, 0, sizeof(*s));
    return 0;
}

int wubu_sb_find(const wubu_sandbox_t *s, const char *name)
{
    if (!s || !name) return -1;
    for (int i = 0; i < s->n; i++)
        if (s->cts[i].present && strcmp(s->cts[i].name, name) == 0) return i;
    return -1;
}

int wubu_sb_create(wubu_sandbox_t *s, const char *name,
                   int64_t cpu_quota_us, int64_t cpu_period_us,
                   uint64_t mem_max, uint32_t pids_max, uint64_t io_max_bps)
{
    if (!s || !name) return -1;
    if (wubu_sb_find(s, name) >= 0) return -1;   /* dup */
    if (s->n >= WUBU_SB_MAX_CONTAINERS) return -1;
    if (cpu_period_us < 0 || cpu_quota_us < 0) return -1;
    wubu_sb_ct_t *c = &s->cts[s->n++];
    memset(c, 0, sizeof(*c));
    strncpy(c->name, name, WUBU_SB_NAME_MAX - 1);
    c->present = 1;
    c->cpu_quota_us = cpu_quota_us;
    c->cpu_period_us = cpu_period_us;
    c->mem_max = mem_max;
    c->pids_max = pids_max;
    c->io_max_bps = io_max_bps;
    c->seccomp_default_deny = 1;   /* default-deny seccomp */
    return 0;
}

int wubu_sb_allow_syscall(wubu_sandbox_t *s, const char *name, int syscall_nr)
{
    if (!s) return -1;
    int i = wubu_sb_find(s, name);
    if (i < 0) return -1;
    wubu_sb_ct_t *c = &s->cts[i];
    for (int k = 0; k < c->n_allow; k++)
        if (c->allow[k] == syscall_nr) return 0;   /* already allowed */
    if (c->n_allow >= WUBU_SB_MAX_SYSCALLS) return -1;
    c->allow[c->n_allow++] = syscall_nr;
    return 0;
}

int wubu_sb_set_default_deny(wubu_sandbox_t *s, const char *name, int deny)
{
    if (!s) return -1;
    int i = wubu_sb_find(s, name);
    if (i < 0) return -1;
    s->cts[i].seccomp_default_deny = deny ? 1 : 0;
    return 0;
}

int wubu_sb_syscall(wubu_sandbox_t *s, const char *name, int syscall_nr)
{
    if (!s) return -1;
    int i = wubu_sb_find(s, name);
    if (i < 0) return -1;
    wubu_sb_ct_t *c = &s->cts[i];
    int allowed = 0;
    for (int k = 0; k < c->n_allow; k++)
        if (c->allow[k] == syscall_nr) { allowed = 1; break; }
    if (!allowed && c->seccomp_default_deny) {
        c->n_blocked++;
        return 0;   /* blocked by seccomp */
    }
    c->n_admitted++;
    return 1;
}

int wubu_sb_account(wubu_sandbox_t *s, const char *name,
                    uint64_t cpu_us, uint64_t mem_bytes,
                    uint32_t pids, uint64_t io_bytes)
{
    if (!s) return -1;
    int i = wubu_sb_find(s, name);
    if (i < 0) return -1;
    wubu_sb_ct_t *c = &s->cts[i];
    c->cpu_used_us += cpu_us;
    c->mem_used += mem_bytes;
    c->pids_used += pids;
    c->io_used_bytes += io_bytes;
    return 0;
}

int wubu_sb_cpu_throttled(const wubu_sandbox_t *s, const char *name)
{
    int i = wubu_sb_find(s, name);
    if (i < 0) return -1;
    const wubu_sb_ct_t *c = &s->cts[i];
    if (c->cpu_quota_us <= 0) return 0;
    return c->cpu_used_us >= (uint64_t)c->cpu_quota_us ? 1 : 0;
}

int wubu_sb_mem_exceeded(const wubu_sandbox_t *s, const char *name)
{
    int i = wubu_sb_find(s, name);
    if (i < 0) return -1;
    const wubu_sb_ct_t *c = &s->cts[i];
    if (c->mem_max == 0) return 0;
    return c->mem_used > c->mem_max ? 1 : 0;
}

int wubu_sb_pids_exceeded(const wubu_sandbox_t *s, const char *name)
{
    int i = wubu_sb_find(s, name);
    if (i < 0) return -1;
    const wubu_sb_ct_t *c = &s->cts[i];
    if (c->pids_max == 0) return 0;
    return c->pids_used > c->pids_max ? 1 : 0;
}

int wubu_sb_io_exceeded(const wubu_sandbox_t *s, const char *name)
{
    int i = wubu_sb_find(s, name);
    if (i < 0) return -1;
    const wubu_sb_ct_t *c = &s->cts[i];
    if (c->io_max_bps == 0) return 0;
    return c->io_used_bytes > c->io_max_bps ? 1 : 0;
}

int wubu_sb_isolated(const wubu_sandbox_t *s, const char *name)
{
    int i = wubu_sb_find(s, name);
    if (i < 0) return -1;
    const wubu_sb_ct_t *c = &s->cts[i];
    /* a sandbox is "isolated" if within all limits and seccomp is default-deny */
    if (wubu_sb_cpu_throttled(s, name) != 0) return 0;
    if (wubu_sb_mem_exceeded(s, name) != 0) return 0;
    if (wubu_sb_pids_exceeded(s, name) != 0) return 0;
    if (wubu_sb_io_exceeded(s, name) != 0) return 0;
    if (!c->seccomp_default_deny) return 0;   /* no syscall filter = not isolated */
    return 1;
}
