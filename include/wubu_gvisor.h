#ifndef WUBU_GVISOR_H
#define WUBU_GVISOR_H

/* wubu_gvisor.h -- RUNTIME #351 gVisor user-kernel isolation.
 *
 * A gVisor-style application-execution sandbox that interposes Linux syscalls
 * in userspace rather than running a separate kernel. Models: an interception
 * table of syscalls, a user-space "Sentry" that services permitted requests,
 * the seccomp-SVP (or seccomp trap) mechanism that traps every syscall into the
 * user-space kernel, and a fail-closed deny-by-default syscall policy. Per
 * #351 the policy is the real work here; the host-kernel exec leg and the OCI
 * image layer are already covered by #340/#341/#342/#349. Deterministic, no
 * stubs. (Distinct from #290's host-kernel kernel isolation.) */

#include <stddef.h>

#define WUBU_GV_MAX_SYSCALLS 300  /* cover x86-64 Linux syscall range */
#define WUBU_GV_NAME_MAX 32

typedef enum {
    WUBU_GV_ACTION_DENY = 0,
    WUBU_GV_ACTION_ADMIT,
} wubu_gv_action_t;

typedef struct {
    int nr;
    char name[WUBU_GV_NAME_MAX];
    wubu_gv_action_t default_action;   /* deny-by-default unless added to admit */
} wubu_gv_syscall_t;

typedef struct {
    wubu_gv_syscall_t table[WUBU_GV_MAX_SYSCALLS];
    int n;                         /* number of registered syscalls */
    int admit[WUBU_GV_MAX_SYSCALLS];  /* 1 if admitted, indexed by nr */
    int n_admitted;
    int n_denied;
    int n_trapped;                 /* syscalls intercepted by seccomp-SVP */
    int seccomp_on;                /* seccomp filter installed */
    int sentry_ready;              /* user-space kernel ready to service */
} wubu_gvisor_t;

/* Initialise the gVisor sandbox: deny-by-default, seccomp off, sentry down. */
int wubu_gv_init(wubu_gvisor_t *gv);

/* Register a syscall by number+name into the interception table. */
int wubu_gv_register_syscall(wubu_gvisor_t *gv, int nr, const char *name);

/* Install the seccomp-SVP filter so all syscalls trap to the user-space
 * kernel. Returns 0/-1. */
int wubu_gv_install_seccomp(wubu_gvisor_t *gv);

/* Bring the user-space Sentry online. */
int wubu_gv_ready_sentry(wubu_gvisor_t *gv);

/* Admit a syscall (it becomes a permitted emulation target). */
int wubu_gv_admit(wubu_gvisor_t *gv, int nr);

/* The user-space kernel services an intercepted syscall. Returns the action
 * the syscall takes: DENY (trapped+denied) or ADMIT (emulated by Sentry). */
int wubu_gv_service(wubu_gvisor_t *gv, int nr, int *action_out);

/* Run a synthetic application syscall stream through the sandbox for testing.
 * Returns count of admitted syscalls (== exits cleanly), -1 on a denied
 * syscall aborting the stream. */
int wubu_gv_run(wubu_gvisor_t *gv, const int *stream, size_t n);

/* Query. */
int wubu_gv_is_admitted(const wubu_gvisor_t *gv, int nr);
int wubu_gv_n_trapped(const wubu_gvisor_t *gv);

#endif
