/* wubu_ro_usr.c -- OMARCHY #289 immutable /usr read-only root.
 * See wubu_ro_usr.h. */
#include "wubu_ro_usr.h"
#include <string.h>

int wubu_ro_init(wubu_ro_usr_t *p) {
    if (!p) return -1;
    memset(p->prefix, 0, sizeof p->prefix);
    p->n = 0;
    /* read-only system dirs */
    const char *ro[] = { "/usr", "/bin", "/lib", "/etc", "/boot", "/sbin", "/opt" };
    const char *rw[] = { "/var", "/tmp", "/home", "/run", "/dev", "/proc", "/sys" };
    for (int i = 0; i < 7; i++) {
        strncpy(p->prefix[p->n], ro[i], 47);
        p->access[p->n++] = WUBU_FS_READONLY;
    }
    for (int i = 0; i < 7; i++) {
        strncpy(p->prefix[p->n], rw[i], 47);
        p->access[p->n++] = WUBU_FS_WRITABLE;
    }
    return 0;
}

wubu_fs_access_t wubu_ro_classify(const wubu_ro_usr_t *p, const char *path) {
    if (!p || !path) return WUBU_FS_READONLY;   /* fail-closed */
    int best = -1, best_len = -1;
    for (int i = 0; i < p->n; i++) {
        size_t plen = strlen(p->prefix[i]);
        /* prefix match at a path boundary (so /usr doesn't match /username) */
        if (strncmp(path, p->prefix[i], plen) == 0 &&
            (path[plen] == '\0' || path[plen] == '/')) {
            if ((int)plen > best_len) { best_len = (int)plen; best = i; }
        }
    }
    if (best < 0) return WUBU_FS_READONLY;   /* unknown path -> read-only (fail-closed) */
    return p->access[best];
}

int wubu_ro_may_write(const wubu_ro_usr_t *p, const char *path) {
    if (!p) return 0;
    return wubu_ro_classify(p, path) == WUBU_FS_WRITABLE ? 1 : 0;
}
