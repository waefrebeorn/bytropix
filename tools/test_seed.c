/* test_seed.c -- the deterministic seed oracle (AC-E13): the same seed
 * yields the same sequence and the same shuffle (rollouts replayable),
 * different seeds diverge, and the unit draws land in [0, 1). */
#include <stdio.h>
#include <string.h>
#include "wubu_seed.h"

int main(void)
{
    int ok = 1;

    /* 1. the same seed -> the same sequence */
    wubu_seed_t a, b;
    wubu_seed_init(&a, 12345);
    wubu_seed_init(&b, 12345);
    for (int i = 0; i < 100; i++)
        if (wubu_seed_next(&a) != wubu_seed_next(&b)) {
            printf("  same-seed sequence diverged at %d FAIL\n", i); ok = 0; break;
        }

    /* 2. the same seed -> the same shuffle (replayable rollouts) */
    int items1[16], items2[16];
    for (int i = 0; i < 16; i++) { items1[i] = i; items2[i] = i; }
    wubu_seed_init(&a, 777); wubu_seed_init(&b, 777);
    wubu_seed_shuffle(&a, items1, 16);
    wubu_seed_shuffle(&b, items2, 16);
    if (memcmp(items1, items2, sizeof items1) != 0) {
        printf("  same-seed shuffle diverged FAIL\n"); ok = 0;
    }
    /* the shuffle is a permutation (1..15 each once) */
    int seen[16] = { 0 };
    for (int i = 0; i < 16; i++) seen[items1[i]]++;
    for (int i = 0; i < 16; i++)
        if (seen[i] != 1) { printf("  shuffle not a permutation FAIL\n"); ok = 0; break; }

    /* 3. different seeds diverge quickly */
    wubu_seed_init(&a, 1); wubu_seed_init(&b, 2);
    int diverged = 0;
    for (int i = 0; i < 8; i++)
        if (wubu_seed_next(&a) != wubu_seed_next(&b)) { diverged = 1; break; }
    if (!diverged) { printf("  different seeds did not diverge FAIL\n"); ok = 0; }

    /* 4. unit draws land in [0, 1) */
    wubu_seed_init(&a, 42);
    int inrange = 1;
    for (int i = 0; i < 1000; i++) {
        double u = wubu_seed_unit(&a);
        if (u < 0.0 || u >= 1.0) { inrange = 0; break; }
    }
    if (!inrange) { printf("  unit draw out of range FAIL\n"); ok = 0; }

    printf("  seed: replayable=%s permutation=%s diverge=%s range=%s  %s\n",
           memcmp(items1, items2, sizeof items1) == 0 ? "yes" : "NO",
           ok ? "yes" : "NO", diverged ? "yes" : "NO", inrange ? "yes" : "NO",
           ok ? "PASS" : "FAIL");
    printf("%s\n", ok ? "ALL SEED TESTS PASSED" : "SEED FAILURES");
    return ok ? 0 : 1;
}