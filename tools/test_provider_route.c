/* tools/test_provider_route.c
 *
 * Local-vs-remote routing with offline fallback (Track B5). Verifies the
 * agent always works:
 *   1. key present + remote allowed -> remote
 *   2. no key -> local (offline floor)
 *   3. key present but remote disallowed -> local
 *   4. neither key nor local -> still local (never a dead remote)
 */
#include "wubu_provider_route.h"
#include <stdio.h>
#include <string.h>

static int checks = 0, fails = 0;
#define CHECK(cond, msg) do { checks++; \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } } while (0)

int main(void)
{
    printf("=== test_provider_route (offline fallback routing) ===\n");

    /* 1. key + allowed -> remote */
    CHECK(wubu_provider_route_decide(1, 1, 1) == WUBU_ROUTE_REMOTE,
          "key present + remote allowed -> remote");

    /* 2. no key -> local floor */
    CHECK(wubu_provider_route_decide(0, 1, 1) == WUBU_ROUTE_LOCAL,
          "no key -> local (offline floor)");

    /* 3. key but remote disallowed -> local */
    CHECK(wubu_provider_route_decide(1, 0, 1) == WUBU_ROUTE_LOCAL,
          "key present but remote disallowed -> local");

    /* 4. neither -> still local, never a dead remote */
    CHECK(wubu_provider_route_decide(0, 1, 0) == WUBU_ROUTE_LOCAL,
          "no key, no local -> still local (never silently remote)");

    /* 5. name mapping */
    CHECK(strcmp(wubu_provider_route_name(WUBU_ROUTE_REMOTE), "remote") == 0,
          "remote name");
    CHECK(strcmp(wubu_provider_route_name(WUBU_ROUTE_LOCAL), "local") == 0,
          "local name");

    printf("=== test_provider_route: %d checks, %d failures ===\n", checks, fails);
    if (fails) return 1;
    printf("=== ALL PROVIDER_ROUTE TESTS PASSED (the agent always works) ===\n");
    return 0;
}
