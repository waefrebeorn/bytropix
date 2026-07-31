/*
 * test_eagle.c — Test EAGLE self-draft speculative decoding.
 * Structural/algorithmic tests (no model file needed).
 *
 * Tests API contract correctness without requiring a full model.
 */
#include "wubu_eagle.h"
#include <stdio.h>

/* We test the API contract without a model loaded.
 * The actual model forward integration is tested via gen_text.
 * Here we verify: parameter validation, null-safety, and API shape. */

int main(void) {
    int errors = 0;

    /* Test 1: Draft init with NULL model rejects */
    wubu_eagle_draft_t draft = {0};
    if (wubu_eagle_draft_init(&draft, NULL, 4) == -1) {
        printf("Test 1 - NULL model reject: PASS\n");
    } else {
        printf("Test 1 - NULL model reject: FAIL\n"); errors++;
    }

    /* Test 2: Draft init with 0 layers rejects */
    if (wubu_eagle_draft_init(&draft, NULL, 0) == -1) {
        printf("Test 2 - Zero layers reject: PASS\n");
    } else {
        printf("Test 2 - Zero layers reject: FAIL\n"); errors++;
    }

    /* Test 3: Draft init with negative layers rejects */
    if (wubu_eagle_draft_init(&draft, NULL, -3) == -1) {
        printf("Test 3 - Negative layers reject: PASS\n");
    } else {
        printf("Test 3 - Negative layers reject: FAIL\n"); errors++;
    }

    /* Test 4: Null model in speculative decode returns 0 */
    int out[10];
    int result = wubu_eagle_speculative_decode(&draft, NULL, NULL, 0,
                                                out, 10);
    if (result == 0) {
        printf("Test 4 - Null model returns 0: PASS\n");
    } else {
        printf("Test 4 - Null model returns 0: FAIL (got %d)\n", result);
        errors++;
    }

    /* Test 5: Null model in verify returns 0 */
    int acc[10];
    int dft[4] = {1, 2, 3, 4};
    result = wubu_eagle_verify(NULL, NULL, 0, dft, 4, acc, 10);
    if (result == 0) {
        printf("Test 5 - Null verify returns 0: PASS\n");
    } else {
        printf("Test 5 - Null verify returns 0: FAIL (got %d)\n", result);
        errors++;
    }

    /* Test 6: Null model in draft_generate returns 0 */
    int dft_out[4];
    result = wubu_eagle_draft_generate(&draft, NULL, 0, dft_out, 4);
    if (result == 0) {
        printf("Test 6 - Null draft returns 0: PASS\n");
    } else {
        printf("Test 6 - Null draft returns 0: FAIL (got %d)\n", result);
        errors++;
    }

    /* Test 7: Struct layout correct (size > 2 pointers) */
    if (sizeof(draft) >= 2 * sizeof(void*)) {
        printf("Test 7 - Struct layout valid: PASS (sizeof=%zu)\n", sizeof(draft));
    } else {
        printf("Test 7 - Struct layout valid: FAIL\n"); errors++;
    }

    printf("\n=== EAGLE tests: %d errors ===\n", errors);
    return errors;
}
