/* wubu_syscall_profile.c -- OMARCHY #277 seccomp-bpf syscall filter.
 * See wubu_syscall_profile.h. Default-deny allowlist, fail-closed. */
#include "wubu_syscall_profile.h"
#include <string.h>

int wubu_scp_init(wubu_syscall_profile_t *p, const char *name) {
    if (!p) return -1;
    memset(p->allow, 0, sizeof p->allow);
    p->n_allow = 0;
    if (name) { strncpy(p->name, name, sizeof p->name - 1); p->name[sizeof p->name - 1] = '\0'; }
    else p->name[0] = '\0';
    return 0;
}

int wubu_scp_allow(wubu_syscall_profile_t *p, int syscall_nr) {
    if (!p) return -1;
    if (syscall_nr < 0) return -1;
    if (p->n_allow >= WUBU_SYSCALL_MAX_ALLOW) return -1;
    /* dedupe */
    for (int i = 0; i < p->n_allow; i++) if (p->allow[i] == syscall_nr) return 0;
    p->allow[p->n_allow++] = syscall_nr;
    return 0;
}

int wubu_scp_allowed(const wubu_syscall_profile_t *p, int syscall_nr) {
    if (!p) return 0;   /* fail-closed */
    if (syscall_nr < 0) return 0;
    for (int i = 0; i < p->n_allow; i++) if (p->allow[i] == syscall_nr) return 1;
    return 0;   /* default-deny */
}

int wubu_scp_count(const wubu_syscall_profile_t *p) {
    return p ? p->n_allow : 0;
}

/* Minimal presets by syscall name (mapped to conventional x86-64 numbers).
 * Keep small and auditable -- these are the allowlists. */
struct { const char *name; int nr; } kSys[] = {
    {"read", 0}, {"write", 1}, {"close", 3}, {"fstat", 5}, {"mmap", 9},
    {"mprotect", 10}, {"munmap", 11}, {"brk", 12}, {"exit", 60},
    {"exit_group", 231}, {"rt_sigaction", 13}, {"rt_sigprocmask", 14},
    {"futex", 202}, {"nanosleep", 35}, {"clock_gettime", 228},
    {"openat", 257}, {"read", 0}, {"write", 1},
    /* network */
    {"socket", 41}, {"connect", 42}, {"accept", 43}, {"sendto", 44},
    {"recvfrom", 45}, {"setsockopt", 54}, {"bind", 49}, {"listen", 50},
};

static int nr_of(const char *name) {
    for (size_t i = 0; i < sizeof kSys / sizeof *kSys; i++)
        if (strcmp(kSys[i].name, name) == 0) return kSys[i].nr;
    return -1;
}

int wubu_scp_preset(wubu_syscall_profile_t *p, const char *preset) {
    if (!p) return -1;
    if (wubu_scp_init(p, preset) != 0) return -1;
    /* "none" = empty default-deny */
    if (strcmp(preset, "none") == 0) return 0;
    /* common core: memory + exit + signals + time */
    const char *core[] = {"read","write","close","fstat","mmap","mprotect",
        "munmap","brk","exit","exit_group","rt_sigaction","rt_sigprocmask",
        "futex","nanosleep","clock_gettime","openat", NULL};
    for (int i = 0; core[i]; i++) wubu_scp_allow(p, nr_of(core[i]));
    if (strcmp(preset, "io") == 0) return 0;         /* core + file io */
    if (strcmp(preset, "network") == 0) {
        const char *net[] = {"socket","connect","accept","sendto","recvfrom",
            "setsockopt","bind","listen", NULL};
        for (int i = 0; net[i]; i++) wubu_scp_allow(p, nr_of(net[i]));
        return 0;
    }
    if (strcmp(preset, "compute") == 0) return 0;    /* core only */
    /* unknown preset: leave as the core default-deny-minus-core (still closed) */
    return 0;
}
