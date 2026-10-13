/* wubu_unit_sandbox.c -- OMARCHY #275 systemd unit sandboxing (ProtectSystem).
 * See wubu_unit_sandbox.h. Policy decision core, fail-closed. */
#include "wubu_unit_sandbox.h"
#include <string.h>

int wubu_unit_sandbox_init_hardened(wubu_unit_sandbox_t *s) {
    if (!s) return -1;
    s->protect_system = WUBU_PS_STRICT;
    s->protect_home = WUBU_PH_READONLY;
    s->private_tmp = 1;
    s->no_new_privileges = 1;
    s->restrict_namespaces = 1;
    return 0;
}

static int eq_ci(const char *a, const char *b) {
    while (*a && *b) {
        char x = *a, y = *b;
        if (x >= 'A' && x <= 'Z') x += 32;
        if (y >= 'A' && y <= 'Z') y += 32;
        if (x != y) return 0;
        a++; b++;
    }
    return *a == '\0' && *b == '\0';
}

int wubu_unit_sandbox_parse(wubu_unit_sandbox_t *s, const char *directive, const char *value) {
    if (!s || !directive || !value) return -1;
    if (eq_ci(directive, "ProtectSystem")) {
        if (eq_ci(value, "no")) s->protect_system = WUBU_PS_NO;
        else if (eq_ci(value, "strict")) s->protect_system = WUBU_PS_STRICT;
        else if (eq_ci(value, "full")) s->protect_system = WUBU_PS_FULL;
        else return -1;
        return 0;
    }
    if (eq_ci(directive, "ProtectHome")) {
        if (eq_ci(value, "no") || eq_ci(value, "false")) s->protect_home = WUBU_PH_NO;
        else if (eq_ci(value, "yes") || eq_ci(value, "true")) s->protect_home = WUBU_PH_YES;
        else if (eq_ci(value, "read-only")) s->protect_home = WUBU_PH_READONLY;
        else if (eq_ci(value, "tmpfs")) s->protect_home = WUBU_PH_TMPFS;
        else return -1;
        return 0;
    }
    if (eq_ci(directive, "PrivateTmp")) {
        s->private_tmp = (eq_ci(value, "yes") || eq_ci(value, "true") || eq_ci(value, "1"));
        return 0;
    }
    if (eq_ci(directive, "NoNewPrivileges")) {
        s->no_new_privileges = (eq_ci(value, "yes") || eq_ci(value, "true") || eq_ci(value, "1"));
        return 0;
    }
    if (eq_ci(directive, "RestrictNamespaces")) {
        s->restrict_namespaces = (eq_ci(value, "yes") || eq_ci(value, "true") || eq_ci(value, "1"));
        return 0;
    }
    return -1;   /* unknown directive */
}

static int starts_with(const char *p, const char *prefix) {
    size_t n = strlen(prefix);
    return strncmp(p, prefix, n) == 0;
}

wubu_fs_access_t wubu_unit_sandbox_fs_access(const wubu_unit_sandbox_t *s, const char *path) {
    if (!s || !path) return WUBU_FS_DENY;   /* fail-closed */

    /* PrivateTmp: /tmp and /var/tmp are an isolated namespace. */
    if (s->private_tmp && (starts_with(path, "/tmp") || starts_with(path, "/var/tmp")))
        return WUBU_FS_ALLOW;   /* private (writable, isolated) */

    /* ProtectHome governs /home, /root, /run/user. */
    int is_home = starts_with(path, "/home") || starts_with(path, "/root") ||
                  starts_with(path, "/run/user");
    if (is_home) {
        switch (s->protect_home) {
            case WUBU_PH_NO:       return WUBU_FS_ALLOW;
            case WUBU_PH_READONLY: return WUBU_FS_READONLY;
            case WUBU_PH_TMPFS:    return WUBU_FS_ALLOW;   /* empty tmpfs, isolated */
            case WUBU_PH_YES:      return WUBU_FS_DENY;    /* inaccessible */
        }
        return WUBU_FS_DENY;
    }

    /* ProtectSystem governs /usr, /boot, /efi, /etc (system dirs). */
    int is_sys = starts_with(path, "/usr") || starts_with(path, "/boot") ||
                 starts_with(path, "/efi") || starts_with(path, "/etc");
    if (is_sys) {
        switch (s->protect_system) {
            case WUBU_PS_NO:     return WUBU_FS_ALLOW;
            case WUBU_PS_STRICT: return WUBU_FS_READONLY;   /* entire fs hierarchy RO */
            case WUBU_PS_FULL:   return WUBU_FS_READONLY;   /* + /usr fully RO */
        }
        return WUBU_FS_DENY;
    }

    /* Not a protected prefix: systemd makes the whole fs RO under
     * ProtectSystem=strict/full, so anything else is read-only. */
    switch (s->protect_system) {
        case WUBU_PS_NO:     return WUBU_FS_ALLOW;
        case WUBU_PS_STRICT:
        case WUBU_PS_FULL:   return WUBU_FS_READONLY;
    }
    return WUBU_FS_READONLY;
}

int wubu_unit_sandbox_may_escalate(const wubu_unit_sandbox_t *s) {
    if (!s) return 0;
    /* NoNewPrivileges blocks setuid/privilege gain; RestrictNamespaces blocks
     * new user/mount namespaces. Escalation allowed only if BOTH are off. */
    return (!s->no_new_privileges && !s->restrict_namespaces) ? 1 : 0;
}
