/* test_ophook.c -- AGENT #151 operator hook applies/restarts/directive.
 * Proves: apply records, restart requires prior apply (fail-closed),
 * directive always applies, sequence numbers monotonic, stats, fail-closed. */
#include "wubu_ophook.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_ophook_t h;
    CHECK(wubu_ophook_init(&h) == 0, "init ok");

    /* directive always applies */
    int d = wubu_ophook_issue(&h, WUBU_OP_DIRECTIVE, "set_lr=0.001");
    CHECK(d == 0 && h.log[d].applied == 1, "directive applied");
    CHECK(h.log[d].seq == 1, "seq 1");

    /* RESTART before APPLY -> rejected (fail-closed) */
    int r = wubu_ophook_issue(&h, WUBU_OP_RESTART, "attn-layer");
    CHECK(h.log[r].applied == 0, "restart before apply -> rejected");
    CHECK(wubu_ophook_applied(&h, "attn-layer") == 0, "not applied yet");

    /* APPLY the component */
    int a = wubu_ophook_issue(&h, WUBU_OP_APPLY, "attn-layer");
    CHECK(h.log[a].applied == 1, "apply applied");
    CHECK(wubu_ophook_applied(&h, "attn-layer") == 1, "now applied");

    /* RESTART after APPLY -> succeeds */
    int r2 = wubu_ophook_issue(&h, WUBU_OP_RESTART, "attn-layer");
    CHECK(h.log[r2].applied == 1, "restart after apply -> succeeds");

    /* RESTART a never-applied component -> still rejected */
    int r3 = wubu_ophook_issue(&h, WUBU_OP_RESTART, "ffn-layer");
    CHECK(h.log[r3].applied == 0, "restart never-applied -> rejected");

    /* monotonic sequence numbers */
    CHECK(h.log[0].seq < h.log[1].seq && h.log[1].seq < h.log[2].seq, "seq monotonic");

    /* stats */
    CHECK(wubu_ophook_applied_count(&h) == 3, "3 applied (directive, apply, restart-after)");
    CHECK(wubu_ophook_rejected_count(&h) == 2, "2 rejected (2 bad restarts)");

    /* fail-closed */
    CHECK(wubu_ophook_issue(&h, WUBU_OP_APPLY, NULL) == -1, "NULL arg -> -1");
    CHECK(wubu_ophook_issue(NULL, WUBU_OP_APPLY, "x") == -1, "NULL hook -> -1");
    CHECK(wubu_ophook_applied(NULL, "x") == 0, "NULL applied -> 0");
    CHECK(wubu_ophook_applied_count(NULL) == 0, "NULL count -> 0");

    printf("\n=== test_ophook: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
