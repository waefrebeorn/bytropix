/* test_output_guard.c -- AGIEDR #306 LLM-output-as-untrusted validation.
 * Proves: safe output passes, hard dangers rejected, soft dangers sanitized
 * under strict, sanitize strips metachars/control chars, fail-closed on NULL. */
#include "wubu_output_guard.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    /* safe output passes */
    wubu_out_result_t r = wubu_output_guard_check("The answer is 42.", WUBU_OUT_STRICT);
    CHECK(r.verdict == WUBU_OUT_PASS, "safe output passes");
    CHECK(r.n_dangers == 0, "safe output 0 dangers");

    /* hard danger -> reject */
    r = wubu_output_guard_check("run $(rm -rf /)", WUBU_OUT_STRICT);
    CHECK(r.verdict == WUBU_OUT_REJECT, "command substitution rejected");
    r = wubu_output_guard_check("cat ../../etc/passwd", WUBU_OUT_STRICT);
    CHECK(r.verdict == WUBU_OUT_REJECT, "path traversal rejected");
    r = wubu_output_guard_check("a && b", WUBU_OUT_STRICT);
    CHECK(r.verdict == WUBU_OUT_REJECT, "shell && rejected");

    /* soft danger -> sanitize under strict, pass under lenient */
    r = wubu_output_guard_check("value | pipe", WUBU_OUT_STRICT);
    CHECK(r.verdict == WUBU_OUT_SANITIZE, "soft danger sanitized under strict");
    r = wubu_output_guard_check("value | pipe", WUBU_OUT_LENIENT);
    CHECK(r.verdict == WUBU_OUT_PASS, "soft danger passes under lenient");

    /* credential pattern flagged */
    r = wubu_output_guard_check("here is the api_key: xyz", WUBU_OUT_STRICT);
    CHECK(r.n_dangers > 0, "credential pattern flagged");

    /* sanitize strips metachars + control chars */
    char dst[128];
    int n = wubu_output_guard_sanitize("a;b|c$d`e\tf", dst, sizeof dst);
    CHECK(n > 0, "sanitize wrote output");
    CHECK(strchr(dst, ';') == NULL, "sanitize removed ;");
    CHECK(strchr(dst, '|') == NULL, "sanitize removed |");
    CHECK(strchr(dst, '$') == NULL, "sanitize removed $");
    CHECK(strchr(dst, '`') == NULL, "sanitize removed `");
    CHECK(strchr(dst, '\t') == NULL, "sanitize removed control char");
    CHECK(strcmp(dst, "abcdef") == 0, "sanitize kept safe chars");

    /* fail-closed on NULL */
    r = wubu_output_guard_check(NULL, WUBU_OUT_STRICT);
    CHECK(r.verdict == WUBU_OUT_REJECT, "NULL output rejected (fail-closed)");
    CHECK(wubu_output_guard_sanitize(NULL, dst, sizeof dst) == -1, "NULL sanitize -> -1");
    CHECK(wubu_output_guard_sanitize("x", NULL, 10) == -1, "NULL dst -> -1");

    printf("\n=== test_output_guard: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
