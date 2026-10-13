/* wubu_protect_kt.c -- OMARCHY #285 ProtectHome + ProtectKernelTunables.
 * See wubu_protect_kt.h. */
#include "wubu_protect_kt.h"
#include <string.h>

int wubu_kt_init(wubu_protect_kt_t *k) {
    if (!k) return -1;
    k->protect_kernel_tunables = 1;
    k->protect_kernel_modules = 1;
    k->protect_home = 1;   /* read-only */
    return 0;
}

int wubu_kt_is_kernel_path(const char *path) {
    if (!path) return 0;
    return strncmp(path, "/proc/sys", 9) == 0 ||
           strncmp(path, "/sys", 4) == 0 ||
           strcmp(path, "/proc/sysrq-trigger") == 0;
}

wubu_kt_access_t wubu_kt_access(const wubu_protect_kt_t *k, const char *path) {
    if (!k || !path) return WUBU_KT_DENY;   /* fail-closed */
    /* kernel tunables */
    if (wubu_kt_is_kernel_path(path)) {
        return k->protect_kernel_tunables ? WUBU_KT_READONLY : WUBU_KT_DENY;
    }
    /* home paths */
    if (strncmp(path, "/home", 5) == 0 || strncmp(path, "/root", 5) == 0 ||
        strncmp(path, "/run/user", 9) == 0) {
        if (k->protect_home == 2) return WUBU_KT_DENY;
        if (k->protect_home == 1) return WUBU_KT_READONLY;
        return WUBU_KT_ALLOW;
    }
    return WUBU_KT_ALLOW;   /* other paths unaffected */
}
