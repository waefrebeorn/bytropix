#ifndef WUBU_DISTROLESS_H
#define WUBU_DISTROLESS_H
#include <stddef.h>

/* wubu_distroless.h -- OMARCHY #282 distroless minimal container images.
 *
 * A distroless image policy: decides whether a file belongs in a minimal
 * container image. Distroless images contain ONLY the application and its
 * runtime dependencies -- no shell, no package manager, no build tools, no
 * docs/manpages. This module classifies a path as included (app + runtime
 * libs) or excluded (shell, pkg-mgr, build tools, docs), enforcing the
 * minimal-attack-surface principle. Fail-closed: unknown paths excluded.
 * Deterministic, real work (no stubs). */

typedef enum {
    WUBU_DISTROLESS_INCLUDE=0,  /* app binary + runtime deps */
    WUBU_DISTROLESS_EXCLUDE,    /* shell, pkg-mgr, build tools, docs */
} wubu_distroless_verdict_t;

/* Classify a path for a distroless image. Returns INCLUDE/EXCLUDE.
 * Fail-closed: NULL/unknown -> EXCLUDE. */
wubu_distroless_verdict_t wubu_distroless_classify(const char *path);

/* Build a minimal image manifest from a list of app files + declared runtime
 * deps: returns how many of the input paths are included. Writes the included
 * paths (by index) into `included` (up to max). */
int wubu_distroless_manifest(const char *const *paths, int n,
                             int *included, int max);

/* Is a path a shell/pkg-manager/build-tool/doc (always excluded)? */
int wubu_distroless_is_excluded(const char *path);

#endif
