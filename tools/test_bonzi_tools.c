/* test_bonzi_tools.c -- UX #240 companion capability-gated tools.
 * Proves: default-deny (no caps -> no tools), grant enables a tool, ungranted
 * tool denied, unknown tool denied, telemetry, fail-closed on NULL. */
#include "wubu_bonzi_tools.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_bonzi_tools_t bt;
    CHECK(wubu_bt_init(&bt) == 0, "init ok");

    /* register tools with required caps */
    CHECK(wubu_bt_register(&bt, "web_search", "net") == 0, "register web_search (cap net)");
    CHECK(wubu_bt_register(&bt, "read_file", "fs") == 0, "register read_file (cap fs)");
    CHECK(wubu_bt_register(&bt, "shell", "exec") == 0, "register shell (cap exec)");

    /* default-deny: no caps -> no tool invocable */
    CHECK(wubu_bt_may_invoke(&bt, "web_search") == 0, "default-deny: web_search denied");
    CHECK(wubu_bt_invoke(&bt, "web_search") == 0, "invoke denied (no cap)");

    /* grant net + fs */
    CHECK(wubu_bt_grant(&bt, "net") == 0, "grant net");
    CHECK(wubu_bt_grant(&bt, "fs") == 0, "grant fs");
    CHECK(wubu_bt_may_invoke(&bt, "web_search") == 1, "web_search allowed (has net)");
    CHECK(wubu_bt_may_invoke(&bt, "read_file") == 1, "read_file allowed (has fs)");
    /* shell still denied (no exec cap) */
    CHECK(wubu_bt_may_invoke(&bt, "shell") == 0, "shell denied (no exec cap)");

    /* invoke allowed tools */
    CHECK(wubu_bt_invoke(&bt, "web_search") == 1, "invoke web_search ok");
    CHECK(wubu_bt_invoke(&bt, "read_file") == 1, "invoke read_file ok");
    CHECK(wubu_bt_invoke(&bt, "shell") == 0, "invoke shell denied");

    /* unknown tool denied */
    CHECK(wubu_bt_may_invoke(&bt, "bogus") == 0, "unknown tool denied");
    CHECK(wubu_bt_invoke(&bt, "bogus") == 0, "invoke unknown denied");

    /* telemetry */
    CHECK(wubu_bt_allowed(&bt) == 2, "2 allowed invocations");
    CHECK(wubu_bt_denied(&bt) == 3, "3 denied (web_search pre-grant, shell, bogus)");

    /* fail-closed */
    CHECK(wubu_bt_may_invoke(NULL, "web_search") == 0, "NULL may_invoke -> 0");
    CHECK(wubu_bt_invoke(NULL, "web_search") == 0, "NULL invoke -> 0");
    CHECK(wubu_bt_grant(NULL, "net") == -1, "NULL grant -> -1");
    CHECK(wubu_bt_register(NULL, "x", "y") == -1, "NULL register -> -1");

    printf("\n=== test_bonzi_tools: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
