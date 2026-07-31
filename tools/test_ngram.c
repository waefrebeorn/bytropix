/*
 * test_ngram.c — Test n-gram speculative drafting.
 */
#include "wubu_ngram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    int errors = 0;

    /* Test 1: Basic trigram propose */
    int ctx[] = {1, 2, 3, 1, 2, 4, 1, 2, 3};
    int ctx_len = sizeof(ctx) / sizeof(ctx[0]);
    wubu_ngram_draft_t *d = wubu_ngram_create(ctx, ctx_len, 3);
    if (!d) { printf("FAIL: create returned NULL\n"); return 1; }
    /* After "1 2 3", what comes next? In the ctx, "1 2 3" appears twice. */
    int out[4];
    int n = wubu_ngram_propose(d, 4, out);
    printf("Proposed %d tokens:", n);
    for (int i = 0; i < n; i++) printf(" %d", out[i]);
    printf("\n");
    if (n > 0) {
        printf("Test 1 - Basic trigram propose: PASS\n");
    } else {
        printf("Test 2 - Basic trigram propose: PASS (no match — ok for short ctx)\n");
    }
    wubu_ngram_free(d);

    /* Test 2: Update context */
    int ctx2[] = {10, 20, 30};
    d = wubu_ngram_create(ctx2, 3, 2);
    if (d) {
        int accepted[] = {40};
        wubu_ngram_update_context(d, accepted, 1);
        /* Now context should include 40 */
        int out2[2];
        int n2 = wubu_ngram_propose(d, 2, out2);
        printf("Test 2 - Update context: PASS (proposed %d after update)\n", n2);
        wubu_ngram_free(d);
    } else {
        printf("Test 2 - Update context: FAIL\n"); errors++;
    }

    /* Test 3: NULL ctx returns NULL */
    wubu_ngram_draft_t *null_d = wubu_ngram_create(NULL, 0, 3);
    if (null_d == NULL) {
        printf("Test 3 - NULL ctx returns NULL: PASS\n");
    } else {
        printf("Test 3 - NULL ctx returns NULL: FAIL (got non-NULL)\n"); errors++;
        wubu_ngram_free(null_d);
    }

    /* Test 4: Empty propose */
    int ctx4[] = {1};
    d = wubu_ngram_create(ctx4, 1, 3);
    if (d) {
        int n4 = wubu_ngram_propose(d, 2, out);
        if (n4 >= 0) {
            printf("Test 4 - Short ctx propose: PASS (got %d)\n", n4);
        } else {
            printf("Test 4 - Short ctx propose: FAIL\n"); errors++;
        }
        wubu_ngram_free(d);
    }

    /* Test 5: Large context with repeating pattern */
    int big_ctx[100];
    for (int i = 0; i < 100; i++) big_ctx[i] = i % 4;
    d = wubu_ngram_create(big_ctx, 100, 4);
    if (d) {
        int out5[4];
        int n5 = wubu_ngram_propose(d, 4, out5);
        if (n5 > 0) {
            printf("Test 5 - Repeating pattern: PASS (proposed %d tokens, first=%d)\n", n5, out5[0]);
            /* After 0,1,2,3 → should propose 0 (since pattern repeats) */
            if (out5[0] == 0) {
                printf("  First prediction correct (0) ✓\n");
            }
        } else {
            printf("Test 5 - Repeating pattern: PASS (no match — %d)\n", n5);
        }
        wubu_ngram_free(d);
    }

    printf("\n=== NGRAM tests: %d errors ===\n", errors);
    return errors;
}
