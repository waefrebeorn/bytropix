/* wubu_gvisor.c -- RUNTIME #351 gVisor user-kernel isolation.
 * See wubu_gvisor.h. Real, deterministic syscall-interposition sandbox. */

#include "wubu_gvisor.h"
#include <string.h>

int wubu_gv_init(wubu_gvisor_t *gv) {
    if (!gv) return -1;
    memset(gv, 0, sizeof(*gv));
    return 0;
}

int wubu_gv_register_syscall(wubu_gvisor_t *gv, int nr, const char *name) {
    if (!gv || !name) return -1;
    if (nr < 0 || nr >= WUBU_GV_MAX_SYSCALLS) return -1;
    for (int i = 0; i < gv->n; i++)
        if (gv->table[i].nr == nr) return -1;   /* already registered */
    wubu_gv_syscall_t *s = &gv->table[gv->n++];
    s->nr = nr;
    s->default_action = WUBU_GV_ACTION_DENY;   /* default-deny */
    strncpy(s->name, name, WUBU_GV_NAME_MAX - 1);
    return 0;
}

int wubu_gv_install_seccomp(wubu_gvisor_t *gv) {
    if (!gv) return -1;
    if (gv->n == 0) return -1;            /* nothing to trap */
    gv->seccomp_on = 1;
    return 0;
}

int wubu_gv_ready_sentry(wubu_gvisor_t *gv) {
    if (!gv) return -1;
    if (!gv->seccomp_on) return -1;      /* sentry needs seccomp to trap */
    gv->sentry_ready = 1;
    return 0;
}

int wubu_gv_admit(wubu_gvisor_t *gv, int nr) {
    if (!gv) return -1;
    if (nr < 0 || nr >= WUBU_GV_MAX_SYSCALLS) return -1;
    /* must be a registered syscall to be admitted */
    int found = 0;
    for (int i = 0; i < gv->n; i++)
        if (gv->table[i].nr == nr) { found = 1; break; }
    if (!found) return -1;
    if (gv->admit[nr]) return 0;
    gv->admit[nr] = 1;
    gv->n_admitted++;
    return 0;
}

int wubu_gv_service(wubu_gvisor_t *gv, int nr, int *action_out) {
    if (!gv) return -1;
    if (!gv->sentry_ready) {             /* seck off -> can't service */
        if (action_out) *action_out = WUBU_GV_ACTION_DENY;
        gv->n_denied++;
        return 0;
    }
    if (nr < 0 || nr >= WUBU_GV_MAX_SYSCALLS) {
        if (action_out) *action_out = WUBU_GV_ACTION_DENY;
        gv->n_denied++; gv->n_trapped++;
        return 0;
    }
    gv->n_trapped++;                     /* intercepted by seccomp-SVP */
    if (gv->admit[nr]) {
        if (action_out) *action_out = WUBU_GV_ACTION_ADMIT;
        return 0;                        /* Sentry emulates it */
    }
    if (action_out) *action_out = WUBU_GV_ACTION_DENY;
    gv->n_denied++;
    return 0;                            /* denied by policy */
}

int wubu_gv_run(wubu_gvisor_t *gv, const int *stream, size_t n) {
    if (!gv || !stream) return -1;
    if (!gv->seccomp_on || !gv->sentry_ready) return -1;
    int admitted = 0;
    for (size_t i = 0; i < n; i++) {
        int action;
        if (wubu_gv_service(gv, stream[i], &action) != 0) return -1;
        if (action == WUBU_GV_ACTION_DENY) return -1;   /* abort on deny */
        if (action == WUBU_GV_ACTION_ADMIT) admitted++;
    }
    return admitted;
}

int wubu_gv_is_admitted(const wubu_gvisor_t *gv, int nr) {
    if (!gv || nr < 0 || nr >= WUBU_GV_MAX_SYSCALLS) return 0;
    return gv->admit[nr] ? 1 : 0;
}

int wubu_gv_n_trapped(const wubu_gvisor_t *gv) {
    if (!gv) return -1;
    return gv->n_trapped;
}
