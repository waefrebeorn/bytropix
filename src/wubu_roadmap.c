/* wubu_roadmap.c -- PATHS #329 AGI roadmap milestone tracker.
 * See wubu_roadmap.h. */
#include "wubu_roadmap.h"
#include <string.h>

int wubu_rm_init(wubu_roadmap_t *r, int current_q) {
    if (!r) return -1;
    memset(r->ms, 0, sizeof r->ms);
    r->n = 0;
    r->current_q = current_q;
    return 0;
}

int wubu_rm_add(wubu_roadmap_t *r, const char *name, int target_q, wubu_rm_status_t status) {
    if (!r || !name || r->n >= WUBU_RM_MAX) return -1;
    if (strlen(name) >= WUBU_RM_NAMELEN) return -1;
    if (status < WUBU_RM_NOT_STARTED || status > WUBU_RM_DONE) return -1;
    strncpy(r->ms[r->n].name, name, WUBU_RM_NAMELEN - 1); r->ms[r->n].name[WUBU_RM_NAMELEN-1] = '\0';
    r->ms[r->n].target_q = target_q;
    r->ms[r->n].status = status;
    r->n++;
    return 0;
}

int wubu_rm_complete(wubu_roadmap_t *r, const char *name) {
    if (!r || !name) return -1;
    for (int i = 0; i < r->n; i++)
        if (strcmp(r->ms[i].name, name) == 0) { r->ms[i].status = WUBU_RM_DONE; return 0; }
    return -1;
}

double wubu_rm_progress(const wubu_roadmap_t *r) {
    if (!r || r->n == 0) return 0.0;
    int done = 0;
    for (int i = 0; i < r->n; i++) if (r->ms[i].status == WUBU_RM_DONE) done++;
    return (double)done / (double)r->n;
}

int wubu_rm_overdue(const wubu_roadmap_t *r) {
    if (!r) return 0;
    int overdue = 0;
    for (int i = 0; i < r->n; i++)
        if (r->ms[i].status != WUBU_RM_DONE && r->ms[i].target_q < r->current_q) overdue++;
    return overdue;
}

int wubu_rm_health(const wubu_roadmap_t *r) {
    if (!r || r->n == 0) return 0;
    int overdue = wubu_rm_overdue(r);
    if (overdue == 0) return 0;                       /* on-track */
    if ((double)overdue > 0.25 * r->n) return 2;      /* behind (>25% overdue) */
    return 1;                                          /* at-risk */
}
