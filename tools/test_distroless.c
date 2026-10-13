/* test_distroless.c -- OMARCHY #282 distroless minimal container images.
 * Proves: app + runtime libs included, shell/pkg-mgr/build-tools/docs
 * excluded, unknown paths excluded (fail-closed), manifest counting, fail-closed. */
#include "wubu_distroless.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    /* app binary + runtime libs included */
    CHECK(wubu_distroless_classify("/app/server") == WUBU_DISTROLESS_INCLUDE, "app binary included");
    CHECK(wubu_distroless_classify("/usr/lib/libc.so") == WUBU_DISTROLESS_INCLUDE, "runtime lib included");
    CHECK(wubu_distroless_classify("/lib/ld.so") == WUBU_DISTROLESS_INCLUDE, "loader included");

    /* shell, pkg-mgr, build tools excluded */
    CHECK(wubu_distroless_classify("/bin/sh") == WUBU_DISTROLESS_EXCLUDE, "shell excluded");
    CHECK(wubu_distroless_classify("/bin/bash") == WUBU_DISTROLESS_EXCLUDE, "bash excluded");
    CHECK(wubu_distroless_classify("/usr/bin/pacman") == WUBU_DISTROLESS_EXCLUDE, "pacman excluded");
    CHECK(wubu_distroless_classify("/usr/bin/make") == WUBU_DISTROLESS_EXCLUDE, "make excluded");
    CHECK(wubu_distroless_classify("/usr/bin/gcc") == WUBU_DISTROLESS_EXCLUDE, "gcc excluded");

    /* docs/manpages/cache excluded */
    CHECK(wubu_distroless_classify("/usr/share/man/x") == WUBU_DISTROLESS_EXCLUDE, "man excluded");
    CHECK(wubu_distroless_classify("/var/cache/x") == WUBU_DISTROLESS_EXCLUDE, "cache excluded");

    /* unknown path excluded (fail-closed, minimal surface) */
    CHECK(wubu_distroless_classify("/opt/random") == WUBU_DISTROLESS_EXCLUDE, "unknown path excluded");
    CHECK(wubu_distroless_classify("/etc/config") == WUBU_DISTROLESS_EXCLUDE, "config excluded (not runtime)");

    /* is_excluded helper */
    CHECK(wubu_distroless_is_excluded("/bin/sh") == 1, "is_excluded shell");
    CHECK(wubu_distroless_is_excluded("/app/server") == 0, "is_excluded app = 0");

    /* manifest: count included from a mixed list */
    const char *paths[] = {"/app/server", "/bin/sh", "/usr/lib/libc.so",
                           "/usr/bin/pacman", "/lib/ld.so", "/usr/share/man/x"};
    int included[8];
    int n = wubu_distroless_manifest(paths, 6, included, 8);
    CHECK(n == 3, "manifest: 3 included (app, libc, ld)");
    CHECK(included[0] == 0 && included[1] == 2 && included[2] == 4, "manifest indices correct");

    /* fail-closed */
    CHECK(wubu_distroless_classify(NULL) == WUBU_DISTROLESS_EXCLUDE, "NULL classify -> exclude");
    CHECK(wubu_distroless_is_excluded(NULL) == 1, "NULL is_excluded -> 1");
    CHECK(wubu_distroless_manifest(NULL, 0, included, 8) == -1, "NULL manifest -> -1");

    printf("\n=== test_distroless: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
