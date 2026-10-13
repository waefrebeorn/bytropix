#ifndef WUBU_SYSCALL_PROFILE_H
#define WUBU_SYSCALL_PROFILE_H
#include <stddef.h>

/* wubu_syscall_profile.h -- OMARCHY #277 seccomp-bpf syscall filter.
 *
 * A minimal-allowlist syscall filter: DEFAULT-DENY every syscall, allow only
 * an explicit whitelist (the "minimal syscall profile per unit"). This is the
 * policy core a seccomp-bpf filter is built from; the actual BPF program is
 * the kernel's job. Each profile is a named set of allowed syscall numbers.
 * Fail-closed: an unknown syscall, a NULL profile, or an uninitialised
 * profile denies. Deterministic, no allocation, real work (no stubs). */

#define WUBU_SYSCALL_MAX_ALLOW 128

typedef struct {
    char name[32];
    int  allow[WUBU_SYSCALL_MAX_ALLOW];
    int  n_allow;
} wubu_syscall_profile_t;

/* Create/initialise a named empty (default-deny) profile. */
int wubu_scp_init(wubu_syscall_profile_t *p, const char *name);
/* Allow a syscall number. Returns 0/-1. Duplicates are no-ops (0). */
int wubu_scp_allow(wubu_syscall_profile_t *p, int syscall_nr);
/* Decide: is this syscall allowed by the profile? Fail-closed 1/0. */
int wubu_scp_allowed(const wubu_syscall_profile_t *p, int syscall_nr);
/* Build a minimal profile from a preset name ("io", "compute", "network",
 * "none"). Unknown preset -> empty (default-deny). Returns 0/-1. */
int wubu_scp_preset(wubu_syscall_profile_t *p, const char *preset);
/* Number of allowed syscalls. */
int wubu_scp_count(const wubu_syscall_profile_t *p);

#endif
