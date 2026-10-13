/* wubu_distroless.c -- OMARCHY #282 distroless minimal container images.
 * See wubu_distroless.h. */
#include "wubu_distroless.h"
#include <string.h>

/* Paths always excluded from a distroless image (shell, pkg-mgr, build tools,
 * docs, package caches). */
static const char *const kExcl[] = {
    "/bin/sh", "/bin/bash", "/bin/dash", "/bin/zsh", "/usr/bin/sh",
    "/bin/pacman", "/usr/bin/pacman", "/bin/apt", "/usr/bin/apt",
    "/bin/dpkg", "/usr/bin/dpkg", "/bin/yum", "/usr/bin/yum",
    "/usr/bin/make", "/usr/bin/gcc", "/usr/bin/cc", "/usr/bin/g++",
    "/usr/bin/ld", "/usr/bin/as", "/usr/bin/ar",
    "/usr/share/man", "/usr/share/doc", "/usr/share/info",
    "/var/cache", "/var/lib/pacman", "/var/lib/apt",
    "/usr/include", "/usr/src",
};

int wubu_distroless_is_excluded(const char *path) {
    if (!path) return 1;   /* fail-closed */
    for (size_t i = 0; i < sizeof kExcl / sizeof *kExcl; i++) {
        size_t l = strlen(kExcl[i]);
        if (strncmp(path, kExcl[i], l) == 0 && (path[l] == '\0' || path[l] == '/'))
            return 1;
    }
    return 0;
}

wubu_distroless_verdict_t wubu_distroless_classify(const char *path) {
    if (!path) return WUBU_DISTROLESS_EXCLUDE;   /* fail-closed */
    if (wubu_distroless_is_excluded(path)) return WUBU_DISTROLESS_EXCLUDE;
    /* include: app binaries and runtime libs (/app, /usr/lib, /lib) */
    if (strncmp(path, "/app", 4) == 0 ||
        strncmp(path, "/usr/lib", 8) == 0 ||
        strncmp(path, "/lib", 4) == 0 ||
        strncmp(path, "/usr/local/lib", 14) == 0)
        return WUBU_DISTROLESS_INCLUDE;
    /* unknown path: excluded (minimal surface, fail-closed) */
    return WUBU_DISTROLESS_EXCLUDE;
}

int wubu_distroless_manifest(const char *const *paths, int n, int *included, int max) {
    if (!paths || !included || n < 0 || max < 0) return -1;
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (wubu_distroless_classify(paths[i]) == WUBU_DISTROLESS_INCLUDE) {
            if (count < max) included[count] = i;
            count++;
        }
    }
    return count;
}
