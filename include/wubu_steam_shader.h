#ifndef WUBU_STEAM_SHADER_H
#define WUBU_STEAM_SHADER_H

/* wubu_steam_shader.h -- Steam shader pre-caching.
 *
 * Models Steam's background shader pre-caching: while a game downloads (or
 * idles), Steam pre-compiles the DXVK/Vulkan shader pipeline states so the
 * first launch doesn't stutter. Models a job queue of shader-pipeline compile
 * tasks with priorities, a worker that drains the queue (compile rate limited),
 * progress tracking, and a cache-hit counter. Deterministic, no stubs. */

#include <stddef.h>
#include <stdint.h>

#define WUBU_SHDR_MAX_JOBS 256
#define WUBU_SHDR_NAME_MAX 48

typedef enum {
    WUBU_SHDR_PENDING = 0,
    WUBU_SHDR_COMPILING,
    WUBU_SHDR_DONE,
    WUBU_SHDR_FAILED
} wubu_shdr_state_t;

typedef struct {
    char name[WUBU_SHDR_NAME_MAX];   /* pipeline id */
    int priority;                    /* higher = compiled first */
    int cost_ms;                     /* compile time cost */
    int done_ms;                     /* compile progress so far */
    wubu_shdr_state_t state;
} wubu_shdr_job_t;

typedef struct {
    wubu_shdr_job_t jobs[WUBU_SHDR_MAX_JOBS];
    int n_jobs;
    int n_done;
    int n_failed;
    int compile_rate_ms;   /* ms of compile work done per work() call */
    int running;           /* pre-caching active */
} wubu_steam_shader_t;

/* Initialise (compile_rate_ms, e.g. 5 ms of work per tick). */
int wubu_shdr_init(wubu_steam_shader_t *sc, int compile_rate_ms);

/* Enqueue a shader-pipeline compile job. Returns 0/-1 (dup/full). */
int wubu_shdr_enqueue(wubu_steam_shader_t *sc, const char *name, int priority,
                      int cost_ms);

/* Mark a job failed (compile error). Returns 0/-1. */
int wubu_shdr_fail(wubu_steam_shader_t *sc, const char *name);

/* One work tick: pick the highest-priority PENDING job, advance its compile
 * progress by compile_rate_ms, complete or continue it. Returns the job index
 * worked on, or -1 when the queue is empty. */
int wubu_shdr_work(wubu_steam_shader_t *sc);

/* Drain the whole queue (repeatedly work until empty). Returns jobs done. */
int wubu_shdr_drain(wubu_steam_shader_t *sc);

/* Query. */
int wubu_shdr_pending(const wubu_steam_shader_t *sc);
int wubu_shdr_done_count(const wubu_steam_shader_t *sc);
wubu_shdr_state_t wubu_shdr_job_state(const wubu_steam_shader_t *sc,
                                      const char *name);

#endif
