/* test_ns_iso.c -- OMARCHY #278 namespace isolation (mount/pid/net/user).
 * Proves: fully-isolated default, directive parsing toggles private/shared,
 * is_private, may_create gated by RestrictNamespaces, isolation score,
 * fail-closed on NULL. */
#include "wubu_ns_iso.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_ns_iso_t n;
    CHECK(wubu_ns_iso_init(&n) == 0, "init ok");
    CHECK(wubu_ns_is_private(&n, WUBU_NS_MOUNT) == 1, "mount private by default");
    CHECK(wubu_ns_is_private(&n, WUBU_NS_PID) == 1, "pid private by default");
    CHECK(wubu_ns_is_private(&n, WUBU_NS_NET) == 1, "net private by default");
    CHECK(wubu_ns_is_private(&n, WUBU_NS_USER) == 1, "user private by default");
    CHECK(wubu_ns_may_create(&n, WUBU_NS_MOUNT) == 0, "cannot create new ns by default");
    CHECK(fabs(wubu_ns_isolation_score(&n) - 1.0) < 1e-9, "full isolation score 1.0");

    /* share the network namespace */
    CHECK(wubu_ns_iso_parse(&n, "PrivateNetwork", "no") == 0, "parse PrivateNetwork=no");
    CHECK(wubu_ns_is_private(&n, WUBU_NS_NET) == 0, "net now shared");
    CHECK(wubu_ns_is_private(&n, WUBU_NS_MOUNT) == 1, "mount still private");
    CHECK(fabs(wubu_ns_isolation_score(&n) - 6.0/7.0) < 1e-9, "score 6/7");

    /* re-privatise */
    CHECK(wubu_ns_iso_parse(&n, "PrivateNetwork", "yes") == 0, "re-privatise net");
    CHECK(wubu_ns_is_private(&n, WUBU_NS_NET) == 1, "net private again");

    /* RestrictNamespaces gates new-namespace creation */
    CHECK(wubu_ns_iso_parse(&n, "RestrictNamespaces", "no") == 0, "parse RestrictNamespaces=no");
    CHECK(wubu_ns_may_create(&n, WUBU_NS_MOUNT) == 1, "may create new ns when unrestricted");
    CHECK(wubu_ns_iso_parse(&n, "RestrictNamespaces", "yes") == 0, "parse RestrictNamespaces=yes");
    CHECK(wubu_ns_may_create(&n, WUBU_NS_MOUNT) == 0, "cannot create new ns when restricted");

    /* PID/user/mount/ipc parsing */
    CHECK(wubu_ns_iso_parse(&n, "PrivatePID", "no") == 0, "parse PrivatePID=no");
    CHECK(wubu_ns_is_private(&n, WUBU_NS_PID) == 0, "pid shared");
    CHECK(wubu_ns_iso_parse(&n, "PrivateUsers", "no") == 0, "parse PrivateUsers=no");
    CHECK(wubu_ns_is_private(&n, WUBU_NS_USER) == 0, "user shared");

    /* unknown directive */
    CHECK(wubu_ns_iso_parse(&n, "Bogus", "yes") == -1, "unknown directive -> -1");

    /* fail-closed on NULL */
    CHECK(wubu_ns_is_private(NULL, WUBU_NS_MOUNT) == 1, "NULL -> isolated (fail-closed)");
    CHECK(wubu_ns_may_create(NULL, WUBU_NS_MOUNT) == 0, "NULL -> cannot create");
    CHECK(fabs(wubu_ns_isolation_score(NULL) - 1.0) < 1e-9, "NULL -> score 1.0");

    printf("\n=== test_ns_iso: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
