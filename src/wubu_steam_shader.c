/* wubu_steam_shader.c -- Steam shader pre-caching.
 * See wubu_steam_shader.h. Real, deterministic shader pre-compile queue. */

#include "wubu_steam_shader.h"
#include <string.h>

int wubu_shdr_init(wubu_steam_shader_t *sc, int compile_rate_ms) {
    if (!sc || compile_rate_ms <= 0) return -1;
    memset(sc, 0, sizeof(*sc));
    sc->compile_rate_ms = compile_rate_ms;
    sc->running = 1;
    return 0;
}

static int find(const wubu_steam_shader_t *sc, const char *name) {
    if (!sc || !name) return -1;
    for (int i = 0; i < sc->n_jobs; i++)
        if (strcmp(sc->jobs[i].name, name) == 0) return i;
    return -1;
}

int wubu_shdr_enqueue(wubu_steam_shader_t *sc, const char *name, int priority,
                      int cost_ms) {
    if (!sc || !name || cost_ms < 0) return -1;
    if (find(sc, name) >= 0) return -1;   /* dup */
    if (sc->n_jobs >= WUBU_SHDR_MAX_JOBS) return -1;
    wubu_shdr_job_t *j = &sc->jobs[sc->n_jobs++];
    memset(j, 0, sizeof(*j));
    strncpy(j->name, name, WUBU_SHDR_NAME_MAX - 1);
    j->priority = priority;
    j->cost_ms = cost_ms;
    j->state = WUBU_SHDR_PENDING;
    return 0;
}

int wubu_shdr_fail(wubu_steam_shader_t *sc, const char *name) {
    if (!sc) return -1;
    int i = find(sc, name);
    if (i < 0) return -1;
    if (sc->jobs[i].state == WUBU_SHDR_DONE || sc->jobs[i].state == WUBU_SHDR_FAILED)
        return -1;   /* already terminal */
    sc->jobs[i].state = WUBU_SHDR_FAILED;
    sc->n_failed++;
    return 0;
}

int wubu_shdr_work(wubu_steam_shader_t *sc) {
    if (!sc || !sc->running) return -1;
    /* pick the highest-priority PENDING (or in-progress COMPILING) job */
    int best = -1;
    for (int i = 0; i < sc->n_jobs; i++) {
        wubu_shdr_job_t *j = &sc->jobs[i];
        if (j->state == WUBU_SHDR_PENDING || j->state == WUBU_SHDR_COMPILING) {
            if (best < 0 || j->priority > sc->jobs[best].priority) best = i;
        }
    }
    if (best < 0) return -1;   /* queue empty */
    wubu_shdr_job_t *j = &sc->jobs[best];
    j->state = WUBU_SHDR_COMPILING;
    j->done_ms += sc->compile_rate_ms;
    if (j->done_ms >= j->cost_ms) {
        j->done_ms = j->cost_ms;
        j->state = WUBU_SHDR_DONE;
        sc->n_done++;
    }
    return best;
}

int wubu_shdr_drain(wubu_steam_shader_t *sc) {
    if (!sc) return -1;
    int worked = 0;
    while (wubu_shdr_work(sc) >= 0) worked++;
    return worked;
}

int wubu_shdr_pending(const wubu_steam_shader_t *sc) {
    if (!sc) return -1;
    int n = 0;
    for (int i = 0; i < sc->n_jobs; i++)
        if (sc->jobs[i].state == WUBU_SHDR_PENDING ||
            sc->jobs[i].state == WUBU_SHDR_COMPILING) n++;
    return n;
}

int wubu_shdr_done_count(const wubu_steam_shader_t *sc) {
    if (!sc) return -1;
    return sc->n_done;
}

wubu_shdr_state_t wubu_shdr_job_state(const wubu_steam_shader_t *sc,
                                      const char *name) {
    int i = find(sc, name);
    if (i < 0) return WUBU_SHDR_PENDING;   /* unknown -> treat as pending? no */
    return sc->jobs[i].state;
}
