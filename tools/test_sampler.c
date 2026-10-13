/* test_sampler.c -- BRAIN #84 sampler DRY anti-repetition for math.
 * Proves: DRY penalty suppresses repeated tokens, breaks repetition loops,
 * penalty grows with repeat count, no penalty below allowed_length, argmax
 * picks the DRY-favored token, fail-closed on NULL/bad vocab. */
#include "wubu_sampler.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_sampler_t s;
    CHECK(wubu_sampler_init(&s, 100, 2.0f, 1.75f, 2.0f) == 0, "init ok");
    CHECK(wubu_sampler_init(&s, 0, 2, 2, 2) == -1, "zero vocab rejected");
    CHECK(wubu_sampler_init(&s, 999999, 2, 2, 2) == -1, "oversize vocab rejected");

    /* commit token 5 three times */
    wubu_sampler_commit(&s, 5);
    wubu_sampler_commit(&s, 5);
    wubu_sampler_commit(&s, 5);
    CHECK(wubu_sampler_commit(&s, -1) == -1, "negative token rejected");
    CHECK(wubu_sampler_commit(&s, 100) == -1, "out-of-range token rejected");

    /* logits: token 5 highest, token 7 second. Without DRY, argmax=5. */
    float logits[100];
    for (int i = 0; i < 100; i++) logits[i] = 1.0f;
    logits[5] = 10.0f;   /* repeated token, highest raw logit */
    logits[7] = 5.0f;    /* fresh token, lower raw logit */
    CHECK(wubu_sampler_argmax(logits, 100) == 5, "without DRY: argmax = repeated token 5");

    /* apply DRY: token 5 (count 3 > allowed 2) gets penalized -> token 7 wins */
    wubu_sampler_apply_dry(&s, logits, 100);
    CHECK(wubu_sampler_argmax(logits, 100) == 7, "with DRY: argmax switches to fresh token 7");
    CHECK(logits[5] < 10.0f, "repeated token logit reduced");

    /* penalty grows with repeat count: commit more, penalty bigger */
    float l1[10], l2[10];
    wubu_sampler_t a, b;
    wubu_sampler_init(&a, 10, 2.0f, 1.75f, 0.0f);
    wubu_sampler_init(&b, 10, 2.0f, 1.75f, 0.0f);
    wubu_sampler_commit(&a, 3);           /* count 1 */
    wubu_sampler_commit(&b, 3); wubu_sampler_commit(&b, 3); wubu_sampler_commit(&b, 3);  /* count 3 */
    for (int i = 0; i < 10; i++) { l1[i] = 8.0f; l2[i] = 8.0f; }
    wubu_sampler_apply_dry(&a, l1, 10);
    wubu_sampler_apply_dry(&b, l2, 10);
    CHECK(l2[3] < l1[3], "more repeats -> stronger penalty");

    /* no penalty below allowed_length */
    wubu_sampler_t c;
    wubu_sampler_init(&c, 10, 2.0f, 1.75f, 5.0f);   /* allowed 5 */
    wubu_sampler_commit(&c, 2); wubu_sampler_commit(&c, 2);   /* count 2 < 5 */
    float l3[10]; for (int i = 0; i < 10; i++) l3[i] = 4.0f;
    wubu_sampler_apply_dry(&c, l3, 10);
    CHECK(l3[2] == 4.0f, "no penalty below allowed_length");

    /* fail-closed on NULL */
    CHECK(wubu_sampler_commit(NULL, 1) == -1, "NULL commit -> -1");
    CHECK(wubu_sampler_apply_dry(NULL, l3, 10) == -1, "NULL apply -> -1");
    CHECK(wubu_sampler_argmax(NULL, 10) == -1, "NULL argmax -> -1");
    CHECK(wubu_sampler_argmax(l3, 0) == -1, "zero n argmax -> -1");

    printf("\n=== test_sampler: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
