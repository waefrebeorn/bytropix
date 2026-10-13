/* wubu_svc_cgroup.c -- KERNEL #15 cgroup v2 per-service via /n/svc.
 * See wubu_svc_cgroup.h. Real, deterministic per-service cgroup controllers. */

#include "wubu_svc_cgroup.h"
#include <string.h>

int wubu_svc_init(wubu_svc_cgroups_t *s)
{
    if (!s) return -1;
    memset(s, 0, sizeof(*s));
    return 0;
}

int wubu_svc_add(wubu_svc_cgroups_t *s, const char *name)
{
    if (!s || !name) return -1;
    if (wubu_svc_find(s, name) >= 0) return -1;   /* dup */
    if (s->n >= WUBU_SVC_MAX) return -1;
    wubu_svc_cgroup_t *c = &s->svcs[s->n++];
    memset(c, 0, sizeof(*c));
    strncpy(c->name, name, WUBU_SVC_NAME_MAX - 1);
    c->present = 1;
    return 0;
}

int wubu_svc_find(const wubu_svc_cgroups_t *s, const char *name)
{
    if (!s || !name) return -1;
    for (int i = 0; i < s->n; i++)
        if (s->svcs[i].present && strcmp(s->svcs[i].name, name) == 0) return i;
    return -1;
}

int wubu_svc_set_cpu(wubu_svc_cgroups_t *s, const char *name,
                     int64_t quota_us, int64_t period_us)
{
    if (!s) return -1;
    int i = wubu_svc_find(s, name);
    if (i < 0) return -1;
    if (period_us <= 0) return -1;          /* period must be positive */
    if (quota_us < 0) return -1;            /* quota non-negative (0 = none) */
    s->svcs[i].cpu_quota_us = quota_us;
    s->svcs[i].cpu_period_us = period_us;
    return 0;
}

int wubu_svc_set_mem(wubu_svc_cgroups_t *s, const char *name, uint64_t max_bytes)
{
    if (!s) return -1;
    int i = wubu_svc_find(s, name);
    if (i < 0) return -1;
    s->svcs[i].mem_max = max_bytes;   /* 0 = unlimited */
    return 0;
}

int wubu_svc_set_io(wubu_svc_cgroups_t *s, const char *name, uint64_t max_bps)
{
    if (!s) return -1;
    int i = wubu_svc_find(s, name);
    if (i < 0) return -1;
    s->svcs[i].io_max_bps = max_bps;  /* 0 = unlimited */
    return 0;
}

int wubu_svc_account(wubu_svc_cgroups_t *s, const char *name,
                     uint64_t cpu_us, uint64_t mem_bytes, uint64_t io_bytes)
{
    if (!s) return -1;
    int i = wubu_svc_find(s, name);
    if (i < 0) return -1;
    s->svcs[i].cpu_used_us += cpu_us;
    s->svcs[i].mem_used += mem_bytes;
    s->svcs[i].io_used_bytes += io_bytes;
    return 0;
}

int wubu_svc_cpu_throttled(const wubu_svc_cgroups_t *s, const char *name)
{
    int i = wubu_svc_find(s, name);
    if (i < 0) return -1;
    const wubu_svc_cgroup_t *c = &s->svcs[i];
    if (c->cpu_quota_us <= 0) return 0;   /* unlimited */
    return c->cpu_used_us >= (uint64_t)c->cpu_quota_us ? 1 : 0;
}

int wubu_svc_mem_exceeded(const wubu_svc_cgroups_t *s, const char *name)
{
    int i = wubu_svc_find(s, name);
    if (i < 0) return -1;
    const wubu_svc_cgroup_t *c = &s->svcs[i];
    if (c->mem_max == 0) return 0;        /* unlimited */
    return c->mem_used > c->mem_max ? 1 : 0;
}

int wubu_svc_io_exceeded(const wubu_svc_cgroups_t *s, const char *name)
{
    int i = wubu_svc_find(s, name);
    if (i < 0) return -1;
    const wubu_svc_cgroup_t *c = &s->svcs[i];
    if (c->io_max_bps == 0) return 0;     /* unlimited */
    return c->io_used_bytes > c->io_max_bps ? 1 : 0;
}
