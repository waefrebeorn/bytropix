/* wubu_oomquar.c -- KERNEL #14 memory quarantine on OOM.
 * See wubu_oomquar.h. Real, deterministic scoped OOM handler. */

#include "wubu_oomquar.h"
#include <string.h>

int wubu_oom_init(wubu_oomquar_t *o)
{
    if (!o) return -1;
    memset(o, 0, sizeof(*o));
    return 0;
}

int wubu_oom_add_scope(wubu_oomquar_t *o, const char *name, uint64_t limit_pages)
{
    if (!o || !name) return -1;
    if (wubu_oom_find_scope(o, name) >= 0) return -1;   /* dup */
    if (o->n_scopes >= WUBU_OOM_MAX_SCOPES) return -1;
    wubu_oom_scope_t *s = &o->scopes[o->n_scopes++];
    memset(s, 0, sizeof(*s));
    strncpy(s->name, name, WUBU_OOM_NAME_MAX - 1);
    s->limit_pages = limit_pages;
    return 0;
}

int wubu_oom_find_scope(const wubu_oomquar_t *o, const char *name)
{
    if (!o || !name) return -1;
    for (int i = 0; i < o->n_scopes; i++)
        if (strcmp(o->scopes[i].name, name) == 0) return i;
    return -1;
}

/* recompute a scope's used_pages from its live tasks */
static void recompute_scope(wubu_oomquar_t *o, int si)
{
    uint64_t used = 0;
    for (int i = 0; i < o->n_tasks; i++)
        if (o->tasks[i].scope == si && !o->tasks[i].quarantined)
            used += o->tasks[i].pages;
    o->scopes[si].used_pages = used;
}

int wubu_oom_add_task(wubu_oomquar_t *o, const char *name, const char *scope_name,
                      uint64_t pages)
{
    if (!o || !name || !scope_name) return -1;
    int si = wubu_oom_find_scope(o, scope_name);
    if (si < 0) return -1;
    if (o->n_tasks >= WUBU_OOM_MAX_TASKS) return -1;
    wubu_oom_task_t *t = &o->tasks[o->n_tasks];
    memset(t, 0, sizeof(*t));
    t->id = o->n_tasks;
    strncpy(t->name, name, WUBU_OOM_NAME_MAX - 1);
    t->scope = si;
    t->pages = pages;
    t->quarantined = 0;
    o->n_tasks++;
    recompute_scope(o, si);
    return t->id;
}

int wubu_oom_over_limit(const wubu_oomquar_t *o, const char *scope_name)
{
    int si = wubu_oom_find_scope(o, scope_name);
    if (si < 0) return -1;
    return o->scopes[si].used_pages > o->scopes[si].limit_pages ? 1 : 0;
}

int wubu_oom_kernel_panicked(const wubu_oomquar_t *o)
{
    return o ? o->kernel_panic : 0;
}

int64_t wubu_oom_handle(wubu_oomquar_t *o, const char *scope_name)
{
    if (!o || !scope_name) return -1;
    int si = wubu_oom_find_scope(o, scope_name);
    if (si < 0) return -1;
    wubu_oom_scope_t *s = &o->scopes[si];
    recompute_scope(o, si);
    if (s->used_pages <= s->limit_pages) return 0;   /* not OOM */

    s->oom_count++;
    int64_t reclaimed = 0;
    /* quarantine the largest reclaimable (non-pinned) live task in THIS scope
     * until under limit. Pinned (kernel-resident) tasks cannot be freed. */
    while (s->used_pages > s->limit_pages) {
        int best = -1;
        uint64_t best_pages = 0;
        for (int i = 0; i < o->n_tasks; i++) {
            if (o->tasks[i].scope != si || o->tasks[i].quarantined) continue;
            if (o->tasks[i].pinned) continue;         /* unreclaimable */
            if (o->tasks[i].pages > best_pages) { best_pages = o->tasks[i].pages; best = i; }
        }
        if (best < 0) {
            /* nothing reclaimable left in this scope -> escalate to panic.
             * The OOM is scoped, but this scope cannot satisfy its limit. */
            s->panic = 1;
            o->kernel_panic = 1;
            break;
        }
        o->tasks[best].quarantined = 1;
        reclaimed += (int64_t)o->tasks[best].pages;
        recompute_scope(o, si);
    }
    o->total_quarantined += (uint64_t)reclaimed;
    return reclaimed;
}
