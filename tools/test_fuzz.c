/* test_fuzz.c -- Theme IX batch 1: the robustness/fuzzing frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_fuzz.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_fuzz (IX batch 1) ===\n");

    /* IX01: mutation produces a different prompt */
    {
        char out[64];
        int n = wubu_fuzz_mutate("tell me a story!", out, 64, 42);
        CHECK(n > 0, "mutated");
        CHECK(strcmp(out, "tell me a story!") != 0, "differs from the seed");
    }

    /* IX02: evasion rate */
    NEAR(wubu_fuzz_evasion(2, 10), 0.2f, 1e-6f);
    NEAR(wubu_fuzz_evasion(0, 0), 0.0f, 1e-6f);

    /* IX03: sensitivity distance */
    {
        int d = -1;
        CHECK(wubu_fuzz_sensitivity("please ignore everything", "ignore", &d) == 1,
              "forbidden found");
        CHECK(d == 7, "distance to the hit");
        wubu_fuzz_sensitivity("a clean prompt", "ignore", &d);
        CHECK(d == -1, "clean -> -1");
    }

    /* IX05: crash validity */
    CHECK(wubu_fuzz_crash_valid(1, 0, 0, 1) == 1, "reachable segv real");
    CHECK(wubu_fuzz_crash_valid(1, 0, 0, 0) == 0, "unreachable ignored");
    CHECK(wubu_fuzz_crash_valid(0, 1, 0, 1) == 0, "oom is environmental");

    /* IX07: divergence */
    NEAR(wubu_fuzz_divergence("abc", "abc"), 0.0f, 1e-6f);
    NEAR(wubu_fuzz_divergence("abc", "xyz"), 1.0f, 1e-6f);

    /* IX08: coverage-guided mutation */
    {
        char out[64];
        uint8_t covered[6] = { 1, 1, 1, 0, 0, 0 };
        wubu_fuzz_cov_mutate("aaa bbb", out, 64, covered, 7);
        CHECK(strcmp(out, "aaa\tbbb") != 0, "uncovered mutated");
    }

    /* IX09: regression gate */
    CHECK(wubu_fuzz_gate(0.5f, 0.2f, 0.1f) == 1, "evasion rose -> gate");
    CHECK(wubu_fuzz_gate(0.25f, 0.2f, 0.1f) == 0, "within tolerance");

    /* IX10: taxonomy */
    {
        int b = -1;
        wubu_fuzz_taxonomy("pretend you are a pirate", &b);
        CHECK(b == 3, "reframed bucketed");
        wubu_fuzz_taxonomy("what is 2+2", &b);
        CHECK(b == 0, "direct bucketed");
    }

    /* IX13: validation */
    CHECK(wubu_fuzz_validate("hello", 100, 1, 1) == 1, "valid passes");
    CHECK(wubu_fuzz_validate("hello\nworld", 100, 1, 1) == 0, "newline rejected");
    CHECK(wubu_fuzz_validate("toolong", 5, 0, 0) == 0, "length rejected");

    /* IX14: seed curation */
    NEAR(wubu_fuzz_seed(1.0f, 1.0f), 1.0f, 1e-5f);
    CHECK(wubu_fuzz_seed(0.2f, 0.0f) < 0.2f, "low yield discounts");

    if (failures == 0) printf("ALL FUZZ TESTS PASSED\n");
    else printf("%d FUZZ FAILURES\n", failures);
    return failures ? 1 : 0;
}
