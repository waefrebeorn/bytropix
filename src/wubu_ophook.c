/* wubu_ophook.c -- AGENT #151 operator hook applies/restarts/directive.
 * See wubu_ophook.h. */
#include "wubu_ophook.h"
#include <string.h>

int wubu_ophook_init(wubu_ophook_t *h) {
    if (!h) return -1;
    memset(h->log, 0, sizeof h->log);
    h->n = 0;
    h->next_seq = 1;
    h->n_applied = 0;
    h->n_rejected = 0;
    return 0;
}

int wubu_ophook_applied(const wubu_ophook_t *h, const char *component) {
    if (!h || !component) return 0;
    for (int i = 0; i < h->n; i++)
        if (h->log[i].cmd == WUBU_OP_APPLY && h->log[i].applied &&
            strcmp(h->log[i].arg, component) == 0)
            return 1;
    return 0;
}

int wubu_ophook_issue(wubu_ophook_t *h, wubu_op_cmd_t cmd, const char *arg) {
    if (!h || !arg || h->n >= WUBU_OPHOOK_MAX) return -1;
    wubu_op_entry_t *e = &h->log[h->n];
    memset(e, 0, sizeof *e);
    e->cmd = cmd;
    strncpy(e->arg, arg, WUBU_OPHOOK_ARGLEN - 1);
    e->seq = h->next_seq++;
    e->applied = 1;
    /* RESTART requires the component to have been APPLIED (fail-closed). */
    if (cmd == WUBU_OP_RESTART && !wubu_ophook_applied(h, arg)) {
        e->applied = 0;
        h->n_rejected++;
        return h->n++;
    }
    h->n_applied++;
    return h->n++;
}

int wubu_ophook_applied_count(const wubu_ophook_t *h) { return h ? h->n_applied : 0; }
int wubu_ophook_rejected_count(const wubu_ophook_t *h) { return h ? h->n_rejected : 0; }
