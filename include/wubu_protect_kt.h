#ifndef WUBU_PROTECT_KT_H
#define WUBU_PROTECT_KT_H

/* wubu_protect_kt.h -- OMARCHY #285 ProtectHome + ProtectKernelTunables.
 *
 * Extends filesystem protection to also cover kernel tunables: /proc/sys,
 * /sys, and /proc/sysrq-trigger (the paths a sandboxed service must not
 * write, or it could reconfigure the kernel). ProtectKernelTunables=yes makes
 * these read-only; ProtectKernelModules=yes blocks module loading; ProtectHome
 * gates /home, /root, /run/user. Fail-closed: an unset policy denies writes
 * to sensitive kernel paths. Deterministic, real work (no stubs). */

typedef enum {
    WUBU_KT_ALLOW=0,   /* writable */
    WUBU_KT_READONLY,  /* read-only */
    WUBU_KT_DENY,      /* inaccessible */
} wubu_kt_access_t;

typedef struct {
    int protect_kernel_tunables;  /* 1 = /proc/sys, /sys read-only */
    int protect_kernel_modules;   /* 1 = no module loading */
    int protect_home;             /* 0=no, 1=read-only, 2=inaccessible */
} wubu_protect_kt_t;

/* Initialise a hardened kernel-protection policy (all on). */
int wubu_kt_init(wubu_protect_kt_t *k);

/* Decide access to a path under the kernel-protection policy. Fail-closed. */
wubu_kt_access_t wubu_kt_access(const wubu_protect_kt_t *k, const char *path);

/* Is a path a kernel-tunable path (/proc/sys, /sys, /proc/sysrq-trigger)? */
int wubu_kt_is_kernel_path(const char *path);

#endif
