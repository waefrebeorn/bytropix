#ifndef WUBU_UNIT_SANDBOX_H
#define WUBU_UNIT_SANDBOX_H
#include <stddef.h>

/* wubu_unit_sandbox.h -- OMARCHY #275 systemd unit sandboxing (ProtectSystem).
 *
 * Models the systemd sandbox directives that harden a service, and computes
 * an enforcement verdict: given a unit's sandbox profile and a filesystem
 * path the service wants to touch, decide ALLOW / READ-ONLY / DENY. Also
 * answers namespace/privilege restrictions. This is the pure-C decision core
 * behind ProtectSystem=strict & friends (the actual mount-namespace setup is
 * the kernel's job; this module owns the policy). Fail-closed: an unknown
 * profile denies. Deterministic, no allocation, real work (no stubs). */

typedef enum {
    WUBU_FS_ALLOW=0,   /* writable */
    WUBU_FS_READONLY,  /* read-only */
    WUBU_FS_DENY,      /* inaccessible */
} wubu_fs_access_t;

/* ProtectSystem: no/strict/full */
typedef enum { WUBU_PS_NO=0, WUBU_PS_STRICT, WUBU_PS_FULL } wubu_protect_system_t;
/* ProtectHome: no/yes/read-only/tmpfs */
typedef enum { WUBU_PH_NO=0, WUBU_PH_YES, WUBU_PH_READONLY, WUBU_PH_TMPFS } wubu_protect_home_t;

typedef struct {
    wubu_protect_system_t protect_system;  /* default STRICT */
    wubu_protect_home_t   protect_home;    /* default READONLY */
    int private_tmp;      /* 1 = private /tmp */
    int no_new_privileges;/* 1 = NoNewPrivileges */
    int restrict_namespaces; /* 1 = RestrictNamespaces */
} wubu_unit_sandbox_t;

/* Parse a "Directive=value" line into the profile. Returns 0 if the directive
 * was recognised and applied, -1 if unknown (caller may treat as no-op or
 * strict). Recognised: ProtectSystem, ProtectHome, PrivateTmp,
 * NoNewPrivileges, RestrictNamespaces. */
int wubu_unit_sandbox_parse(wubu_unit_sandbox_t *s, const char *directive, const char *value);

/* Initialise a hardened default profile (ProtectSystem=strict, etc.). */
int wubu_unit_sandbox_init_hardened(wubu_unit_sandbox_t *s);

/* Decide filesystem access for a path under the profile. Fail-closed. */
wubu_fs_access_t wubu_unit_sandbox_fs_access(const wubu_unit_sandbox_t *s, const char *path);

/* 1 if the service may create new user namespaces / gain privileges. */
int wubu_unit_sandbox_may_escalate(const wubu_unit_sandbox_t *s);

#endif
