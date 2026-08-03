/* test_fmt.c -- the format-constraint reward checker (the Atropos
 * Answer-Format env): the binary format rewards decoupled from the
 * semantics. */
#include <stdio.h>
#include <string.h>
#include "wubu_fmt.h"

int main(void)
{
    int ok = 1;

    /* JSON: valid object, invalid, balanced-bracket edge */
    if (!wubu_fmt_check(WUBU_FMT_JSON, "{\"a\": [1, 2, 3]}", 0, NULL)) {
        printf("  json valid FAIL\n"); ok = 0;
    }
    if (wubu_fmt_check(WUBU_FMT_JSON, "{\"a\": [1, 2}", 0, NULL)) {
        printf("  json unbalanced FAIL\n"); ok = 0;
    }
    if (wubu_fmt_check(WUBU_FMT_JSON, "not json at all", 0, NULL)) {
        printf("  json nonsense FAIL\n"); ok = 0;
    }
    if (!wubu_fmt_check(WUBU_FMT_JSON, "[{\"escaped\": \"\\\"quote\\\"\"}]", 0, NULL)) {
        printf("  json escaped-quote FAIL\n"); ok = 0;
    }

    /* the <think> delimiter: strict open+close, exactly one */
    if (!wubu_fmt_check(WUBU_FMT_THINK, "<think>compute</think>answer", 0, NULL)) {
        printf("  think valid FAIL\n"); ok = 0;
    }
    if (wubu_fmt_check(WUBU_FMT_THINK, "no think tags", 0, NULL)) {
        printf("  think missing FAIL\n"); ok = 0;
    }
    if (wubu_fmt_check(WUBU_FMT_THINK, "<think>open only", 0, NULL)) {
        printf("  think unclosed FAIL\n"); ok = 0;
    }

    /* lengths + the prefix */
    if (!wubu_fmt_check(WUBU_FMT_LEN_MAX, "abc", 5, NULL)) {
        printf("  len-max FAIL\n"); ok = 0;
    }
    if (wubu_fmt_check(WUBU_FMT_LEN_MAX, "abcdef", 5, NULL)) {
        printf("  len-max over FAIL\n"); ok = 0;
    }
    if (!wubu_fmt_check(WUBU_FMT_PREFIX, "## Summary: done", 0, "## Summary")) {
        printf("  prefix FAIL\n"); ok = 0;
    }
    if (wubu_fmt_check(WUBU_FMT_PREFIX, "Summary: done", 0, "## Summary")) {
        printf("  prefix wrong FAIL\n"); ok = 0;
    }

    /* the combined reward: 2 of 3 held */
    int types[] = { WUBU_FMT_THINK, WUBU_FMT_LEN_MAX, WUBU_FMT_JSON };
    int limits[] = { 0, 200, 0 };
    float r = wubu_fmt_reward(types, 3, "<think>x</think>{\"ok\": 1}",
                              limits, NULL);
    if (r < 0.66f || r > 0.67f) { printf("  combined reward %.2f FAIL\n", r); ok = 0; }
    int limits2[] = { 0, 5, 0 };
    float r2 = wubu_fmt_reward(types, 3, "plain text", limits2, NULL);
    if (r2 != 0.0f) { printf("  combined all-fail %.2f FAIL\n", r2); ok = 0; }

    printf("  format checker: json/think/len/prefix/reward  %s\n", ok ? "PASS" : "FAIL");
    printf("%s\n", ok ? "ALL FMT TESTS PASSED" : "FMT FAILURES");
    return ok ? 0 : 1;
}
