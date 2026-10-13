/* test_landlock.c -- OMARCHY #276 Landlock unprivileged filesystem sandbox.
 * Proves: default-deny, path-subtree scoping with boundary, per-right grants,
 * multiple rules, fail-closed on NULL, ungoverned right denied. */
#include "wubu_landlock.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_landlock_t l;
    CHECK(wubu_landlock_init(&l) == 0, "init ok");

    /* default-deny: nothing granted */
    CHECK(wubu_landlock_check(&l, "/etc/passwd", WUBU_LL_READ) == 0, "default-deny read");
    CHECK(wubu_landlock_check(&l, "/home/wubu/x", WUBU_LL_WRITE) == 0, "default-deny write");

    /* grant read under /etc */
    CHECK(wubu_landlock_allow(&l, "/etc", WUBU_LL_READ) == 0, "allow read /etc");
    CHECK(wubu_landlock_check(&l, "/etc/passwd", WUBU_LL_READ) == 1, "read /etc/passwd allowed");
    CHECK(wubu_landlock_check(&l, "/etc", WUBU_LL_READ) == 1, "read /etc (exact) allowed");
    CHECK(wubu_landlock_check(&l, "/etc/ssl/cert.pem", WUBU_LL_READ) == 1, "read nested allowed");
    /* boundary: /etcfoo is NOT within /etc */
    CHECK(wubu_landlock_check(&l, "/etcfoo", WUBU_LL_READ) == 0, "boundary: /etcfoo not within /etc");
    /* write NOT granted (only read) */
    CHECK(wubu_landlock_check(&l, "/etc/passwd", WUBU_LL_WRITE) == 0, "write not granted");
    /* other subtree denied */
    CHECK(wubu_landlock_check(&l, "/root/x", WUBU_LL_READ) == 0, "other subtree denied");

    /* grant read+write under /home/wubu */
    CHECK(wubu_landlock_allow(&l, "/home/wubu", WUBU_LL_READ | WUBU_LL_WRITE) == 0, "allow rw /home/wubu");
    CHECK(wubu_landlock_check(&l, "/home/wubu/doc.txt", WUBU_LL_READ) == 1, "rw: read allowed");
    CHECK(wubu_landlock_check(&l, "/home/wubu/doc.txt", WUBU_LL_WRITE) == 1, "rw: write allowed");
    CHECK(wubu_landlock_check(&l, "/home/wubu/doc.txt", WUBU_LL_EXECUTE) == 0, "rw: execute not granted");

    /* grant execute under /opt/bin */
    CHECK(wubu_landlock_allow(&l, "/opt/bin", WUBU_LL_EXECUTE | WUBU_LL_READ) == 0, "allow exec /opt/bin");
    CHECK(wubu_landlock_check(&l, "/opt/bin/tool", WUBU_LL_EXECUTE) == 1, "exec /opt/bin/tool allowed");
    CHECK(wubu_landlock_check(&l, "/opt/bin/tool", WUBU_LL_READ) == 1, "read /opt/bin/tool allowed");

    /* trailing-slash subtree normalised */
    CHECK(wubu_landlock_allow(&l, "/srv/data/", WUBU_LL_READ) == 0, "allow /srv/data/ (trailing slash)");
    CHECK(wubu_landlock_check(&l, "/srv/data/x", WUBU_LL_READ) == 1, "normalised subtree matches");

    /* empty path rejected */
    CHECK(wubu_landlock_allow(&l, "", WUBU_LL_READ) == -1, "empty path rejected");

    /* fail-closed on NULL */
    CHECK(wubu_landlock_check(NULL, "/etc", WUBU_LL_READ) == 0, "NULL ruleset denies");
    CHECK(wubu_landlock_check(&l, NULL, WUBU_LL_READ) == 0, "NULL path denies");

    printf("\n=== test_landlock: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
