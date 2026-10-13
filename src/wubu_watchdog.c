/* wubu_watchdog.c -- KERNEL #20 watchdog restarts a hung subsystem.
 * See wubu_watchdog.h. */
#include "wubu_watchdog.h"
#include <string.h>

int wubu_wd_init(wubu_watchdog_t *w) {
    if (!w) return -1;
    memset(w->subs, 0, sizeof w->subs);
    w->n = 0;
    w->now = 0;
    return 0;
}

int wubu_wd_register(wubu_watchdog_t *w, const char *name, uint64_t timeout) {
    if (!w || !name || w->n >= WUBU_WD_MAX || timeout == 0) return -1;
    if (strlen(name) >= 24) return -1;
    wubu_wd_sub_t *s = &w->subs[w->n];
    strncpy(s->name, name, 23); s->name[23] = '\0';
    s->last_kick = w->now;
    s->timeout = timeout;
    s->state = WUBU_WD_HEALTHY;
    s->n_restarts = 0;
    w->n++;
    return 0;
}

static wubu_wd_sub_t *find(wubu_watchdog_t *w, const char *name) {
    for (int i = 0; i < w->n; i++)
        if (strcmp(w->subs[i].name, name) == 0) return &w->subs[i];
    return NULL;
}

int wubu_wd_kick(wubu_watchdog_t *w, const char *name) {
    if (!w || !name) return -1;
    wubu_wd_sub_t *s = find(w, name);
    if (!s) return -1;
    s->last_kick = w->now;
    if (s->state == WUBU_WD_HUNG) s->state = WUBU_WD_HEALTHY;   /* recovered */
    return 0;
}

int wubu_wd_tick(wubu_watchdog_t *w, uint64_t n_ticks) {
    if (!w) return -1;
    w->now += n_ticks;
    int restarted = 0;
    for (int i = 0; i < w->n; i++) {
        wubu_wd_sub_t *s = &w->subs[i];
        if (s->state == WUBU_WD_RESTARTED) continue;   /* already handled */
        if (w->now - s->last_kick > s->timeout) {
            s->state = WUBU_WD_HUNG;
            /* scoped restart: reset this subsystem's heartbeat (not the kernel) */
            s->state = WUBU_WD_RESTARTED;
            s->last_kick = w->now;
            s->n_restarts++;
            restarted++;
        }
    }
    return restarted;
}

int wubu_wd_state_of(const wubu_watchdog_t *w, const char *name) {
    if (!w || !name) return -1;
    for (int i = 0; i < w->n; i++)
        if (strcmp(w->subs[i].name, name) == 0) return (int)w->subs[i].state;
    return -1;
}

int wubu_wd_total_restarts(const wubu_watchdog_t *w) {
    if (!w) return 0;
    int total = 0;
    for (int i = 0; i < w->n; i++) total += w->subs[i].n_restarts;
    return total;
}
