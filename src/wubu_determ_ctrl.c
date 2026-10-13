/* wubu_determ_ctrl.c -- AGIEDR #305 deterministic controls outside model plane.
 * See wubu_determ_ctrl.h. */
#include "wubu_determ_ctrl.h"
#include <string.h>

int wubu_dc_init(wubu_determ_ctrl_t *g) {
    if (!g) return -1;
    memset(g->controls, 0, sizeof g->controls);
    g->n = 0;
    g->n_allowed = 0;
    g->n_denied = 0;
    g->last_denied[0] = '\0';
    return 0;
}

int wubu_dc_add(wubu_determ_ctrl_t *g, const char *name, wubu_ctrl_fn fn) {
    if (!g || !name || !fn || g->n >= WUBU_CTRL_MAX) return -1;
    if (strlen(name) >= WUBU_CTRL_NAMELEN) return -1;
    g->controls[g->n].fn = fn;
    strncpy(g->controls[g->n].name, name, WUBU_CTRL_NAMELEN - 1);
    g->controls[g->n].name[WUBU_CTRL_NAMELEN - 1] = '\0';
    g->n++;
    return 0;
}

int wubu_dc_check(wubu_determ_ctrl_t *g, const void *action, void *ctx) {
    if (!g) return 0;   /* fail-closed */
    for (int i = 0; i < g->n; i++) {
        if (!g->controls[i].fn(action, ctx)) {
            g->n_denied++;
            strncpy(g->last_denied, g->controls[i].name, WUBU_CTRL_NAMELEN - 1);
            g->last_denied[WUBU_CTRL_NAMELEN - 1] = '\0';
            return 0;   /* first failing control denies */
        }
    }
    g->n_allowed++;
    return 1;
}

const char *wubu_dc_last_denied(const wubu_determ_ctrl_t *g) {
    return g ? g->last_denied : "";
}

int wubu_dc_allowed(const wubu_determ_ctrl_t *g) { return g ? g->n_allowed : 0; }
int wubu_dc_denied(const wubu_determ_ctrl_t *g) { return g ? g->n_denied : 0; }
