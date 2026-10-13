/* wubu_ctr_rt.c -- OMARCHY #291 containerd-style high-level runtime.
 * See wubu_ctr_rt.h. */
#include "wubu_ctr_rt.h"
#include <string.h>

int wubu_ctr_init(wubu_ctr_rt_t *rt) {
    if (!rt) return -1;
    memset(rt->tasks, 0, sizeof rt->tasks);
    rt->n_tasks = 0;
    rt->n_images = 0;
    rt->n_snapshots = 0;
    return 0;
}

int wubu_ctr_pull_image(wubu_ctr_rt_t *rt) {
    if (!rt) return -1;
    rt->n_images++;
    rt->n_snapshots++;   /* pulling prepares a snapshot */
    return 0;
}

int wubu_ctr_create(wubu_ctr_rt_t *rt, const char *name) {
    if (!rt || !name || rt->n_tasks >= WUBU_CTR_MAX) return -1;
    if (rt->n_snapshots <= 0) return -1;   /* need a snapshot to create from */
    if (strlen(name) >= 48) return -1;
    wubu_ctr_task_t *t = &rt->tasks[rt->n_tasks];
    strncpy(t->name, name, 47); t->name[47] = '\0';
    t->state = WUBU_CTR_CREATED;
    t->has_snapshot = 1;
    rt->n_tasks++;
    return 0;
}

static wubu_ctr_task_t *find(wubu_ctr_rt_t *rt, const char *name) {
    for (int i = 0; i < rt->n_tasks; i++)
        if (strcmp(rt->tasks[i].name, name) == 0) return &rt->tasks[i];
    return NULL;
}

int wubu_ctr_start(wubu_ctr_rt_t *rt, const char *name) {
    if (!rt || !name) return -1;
    wubu_ctr_task_t *t = find(rt, name);
    if (!t || t->state != WUBU_CTR_CREATED) return -1;   /* must be CREATED */
    t->state = WUBU_CTR_RUNNING;
    return 0;
}

int wubu_ctr_stop(wubu_ctr_rt_t *rt, const char *name) {
    if (!rt || !name) return -1;
    wubu_ctr_task_t *t = find(rt, name);
    if (!t || t->state != WUBU_CTR_RUNNING) return -1;   /* must be RUNNING */
    t->state = WUBU_CTR_STOPPED;
    return 0;
}

int wubu_ctr_delete(wubu_ctr_rt_t *rt, const char *name) {
    if (!rt || !name) return -1;
    wubu_ctr_task_t *t = find(rt, name);
    if (!t) return -1;
    if (t->state == WUBU_CTR_RUNNING) return -1;   /* cannot delete a running task */
    if (t->state == WUBU_CTR_DELETED) return -1;   /* already deleted */
    t->state = WUBU_CTR_DELETED;
    if (t->has_snapshot && rt->n_snapshots > 0) rt->n_snapshots--;
    return 0;
}

int wubu_ctr_state_of(const wubu_ctr_rt_t *rt, const char *name) {
    if (!rt || !name) return -1;
    for (int i = 0; i < rt->n_tasks; i++)
        if (strcmp(rt->tasks[i].name, name) == 0) return (int)rt->tasks[i].state;
    return -1;
}
