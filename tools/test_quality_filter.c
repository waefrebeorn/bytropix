/* test_quality_filter.c — BRAIN #245: quality filter forward verified.
 *
 * Measured assertions on wubu_quality_score + wubu_quality_accept:
 *   1. empty/NULL -> 0.
 *   2. score always in [0,1].
 *   3. real prose scores high (> 0.6).
 *   4. numeric spam scores low (< prose).
 *   5. gibberish/repeated-char scores lower than real prose.
 *   6. accept boundary: accept iff score >= min_score.
 *   7. prose accepted at a reasonable threshold; spam rejected.
 */
#include "wubu_quality_filter.h"
#include <stdio.h>
#include <string.h>

static int g_run = 0, g_pass = 0;
#define T(cond, msg) do { \
    g_run++; \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

int main(void) {
    printf("=== test_quality_filter (document quality filter, #245) ===\n");

    /* 1: empty/NULL. */
    T(wubu_quality_score("") == 0.0, "empty -> 0");
    T(wubu_quality_score(NULL) == 0.0, "NULL -> 0");

    const char *prose =
        "The quick brown fox jumps over the lazy dog while the sun sets "
        "slowly behind the distant mountains and the river flows onward "
        "toward the quiet valley where the villagers rest after a long day.";
    const char *spam = "1234567890 1234567890 1234567890 1234567890 1234567890";
    const char *gibberish = "xkcd qwrtz plmnb vcxzs dfghj kjhgf mnbvc xzlkj";

    double sp = wubu_quality_score(prose);
    double ss = wubu_quality_score(spam);
    double sg = wubu_quality_score(gibberish);

    /* 2: in [0,1]. */
    T(sp >= 0.0 && sp <= 1.0 && ss >= 0.0 && ss <= 1.0, "scores in [0,1]");

    /* 3: prose high. */
    T(sp > 0.6, "real prose scores high (> 0.6)");

    /* 4: spam lower than prose. */
    T(ss < sp, "numeric spam scores below prose");

    /* 5: gibberish below prose. */
    T(sg < sp, "gibberish scores below prose");

    /* 6+7: accept boundary + realistic accept/reject. */
    double th = 0.5;
    T(wubu_quality_accept(prose, th) == 1, "prose accepted at 0.5");
    T(wubu_quality_accept(spam, th) == 0, "spam rejected at 0.5");
    /* boundary: accept iff score >= min. */
    T(wubu_quality_accept(prose, sp) == 1, "accept at score==itself (>=)");
    T(wubu_quality_accept(prose, sp + 1e-9) == 0, "reject just above its score");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
