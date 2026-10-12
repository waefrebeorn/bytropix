/* tools/test_tool_cap.c
 *
 * Capability-gated tool dispatch. Verifies AD01 wraps every tool call:
 *   1. a tool bound to a subtree runs for a caller who owns it
 *   2. the same tool is DENIED (-2) for a caller whose subtree doesn't own it
 *   3. an UNBOUND tool is deny-by-default (denied even for the owner)
 *   4. the handler is never invoked on a denial
 */
#include "wubu_tool_cap.h"
#include "wubu_tooluse.h"
#include <string.h>
#include <stdio.h>

static int checks = 0, fails = 0;
#define CHECK(cond, msg) do { checks++; \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } } while (0)

static int g_handler_calls = 0;
static int echo_handler(void *ctx, const char *args, char *out, int cap) {
    (void)ctx;
    g_handler_calls++;
    return snprintf(out, cap, "ran:%s", args ? args : "");
}

int main(void)
{
    printf("=== test_tool_cap (capability-gated dispatch) ===\n");

    wubu_tool_registry_t reg;
    wubu_tool_registry_init(&reg);
    wubu_tool_register(&reg, "screenshot", "capture the screen", "{}");
    wubu_tool_set_handler(&reg, "screenshot", echo_handler, NULL);

    wubu_tool_cap_t caps = {0};
    CHECK(wubu_tool_cap_bind(&caps, "screenshot", "/n/companion/tools") == 0,
          "bind screenshot to /n/companion/tools");

    char out[128];

    /* 1. owner may call */
    g_handler_calls = 0;
    int rc = wubu_tool_cap_call(&caps, &reg, "/n/companion", "screenshot",
                                "full", out, sizeof out);
    CHECK(rc > 0 && strstr(out, "ran:full"), "owner call runs the handler");
    CHECK(g_handler_calls == 1, "handler invoked once for the owner");

    /* 2. non-owner is denied, handler NOT invoked */
    g_handler_calls = 0;
    rc = wubu_tool_cap_call(&caps, &reg, "/n/other", "screenshot",
                            "full", out, sizeof out);
    CHECK(rc == -2, "non-owner call is DENIED (-2)");
    CHECK(g_handler_calls == 0, "handler NOT invoked on a denial");

    /* 3. unbound tool is deny-by-default */
    wubu_tool_register(&reg, "unbound_tool", "no cap bound", "{}");
    wubu_tool_set_handler(&reg, "unbound_tool", echo_handler, NULL);
    g_handler_calls = 0;
    rc = wubu_tool_cap_call(&caps, &reg, "/n/companion", "unbound_tool",
                            "x", out, sizeof out);
    CHECK(rc == -2, "unbound tool is deny-by-default (denied)");
    CHECK(g_handler_calls == 0, "handler NOT invoked for an unbound tool");

    /* 4. ancestor subtree also owns the path (Styx namespace semantics) */
    g_handler_calls = 0;
    rc = wubu_tool_cap_call(&caps, &reg, "/n", "screenshot", "full",
                            out, sizeof out);
    CHECK(rc > 0 && g_handler_calls == 1, "ancestor subtree /n may call");

    printf("=== test_tool_cap: %d checks, %d failures ===\n", checks, fails);
    if (fails) return 1;
    printf("=== ALL TOOL_CAP TESTS PASSED (capability-gated agency) ===\n");
    return 0;
}
