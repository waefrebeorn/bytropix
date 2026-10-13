#ifndef WUBU_CTR_RT_H
#define WUBU_CTR_RT_H

/* wubu_ctr_rt.h -- OMARCHY #291 containerd-style high-level runtime.
 *
 * A containerd-style container runtime: manages the container lifecycle
 * (image -> snapshot -> task -> running -> stopped -> deleted) via a small
 * state machine. Enforces valid transitions (can't start a task that isn't
 * created, can't delete a running task without stopping first). Tracks images,
 * snapshots, and tasks by name. Deterministic, fixed-size, real work (no stubs). */

typedef enum {
    WUBU_CTR_CREATED=0,   /* snapshot created, task not started */
    WUBU_CTR_RUNNING,     /* task is running */
    WUBU_CTR_STOPPED,     /* task stopped, resources held */
    WUBU_CTR_DELETED,     /* task + snapshot removed */
} wubu_ctr_state_t;

#define WUBU_CTR_MAX 32

typedef struct {
    char name[48];
    wubu_ctr_state_t state;
    int has_snapshot;
} wubu_ctr_task_t;

typedef struct {
    wubu_ctr_task_t tasks[WUBU_CTR_MAX];
    int n_tasks;
    int n_images;
    int n_snapshots;
} wubu_ctr_rt_t;

/* Initialise. */
int wubu_ctr_init(wubu_ctr_rt_t *rt);

/* Pull an image (increments image count). Returns 0/-1. */
int wubu_ctr_pull_image(wubu_ctr_rt_t *rt);

/* Create a task from a snapshot (CREATED). Requires a prior pull+snapshot.
 * Returns 0/-1. */
int wubu_ctr_create(wubu_ctr_rt_t *rt, const char *name);

/* Start a task (CREATED -> RUNNING). Returns 0/-1. */
int wubu_ctr_start(wubu_ctr_rt_t *rt, const char *name);

/* Stop a task (RUNNING -> STOPPED). Returns 0/-1. */
int wubu_ctr_stop(wubu_ctr_rt_t *rt, const char *name);

/* Delete a task (STOPPED -> DELETED). Cannot delete a RUNNING task.
 * Returns 0/-1. */
int wubu_ctr_delete(wubu_ctr_rt_t *rt, const char *name);

/* Current state of a task. Returns -1 if unknown. */
int wubu_ctr_state_of(const wubu_ctr_rt_t *rt, const char *name);

#endif
