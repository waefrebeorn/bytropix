/*
 * test_hive.c — Test wubu_hive (linked fixed blocks + skipfield + freelist).
 *
 * Tests cover:
 *   1. Basic create/destroy
 *   2. Insert into first block
 *   3. Insert with block overflow (new block allocated)
 *   4. Erase and slot reuse
 *   5. Find existing and non-existing values
 *   6. Iterate over live slots (skips free)
 *   7. Size tracking accuracy
 *   8. Pointer stability (same pointer after insert/erase cycles)
 */

#include "wubu_hive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int errors = 0;

static void check(int cond, const char *msg) {
    if (!cond) {
        printf("FAIL: %s\n", msg);
        errors++;
    } else {
        printf("PASS: %s\n", msg);
    }
}

/* Test 1: Create and destroy */
void test_create_destroy(void) {
    wubu_hive_t *h = wubu_hive_create(4);
    check(h != NULL, "hive create returns non-NULL");
    check(wubu_hive_size(h) == 0, "new hive has size 0");
    check(wubu_hive_blocks(h) == 1, "new hive has 1 block");
    check(wubu_hive_block_cap(h) == 4, "block cap is 4");
    wubu_hive_destroy(h);
    printf("  test_create_destroy: done\n");
}

/* Test 2: Basic insert */
void test_insert(void) {
    wubu_hive_t *h = wubu_hive_create(4);
    int a = 10, b = 20, c = 30;

    check(wubu_hive_insert(h, &a) == 0, "insert a");
    check(wubu_hive_insert(h, &b) == 0, "insert b");
    check(wubu_hive_insert(h, &c) == 0, "insert c");
    check(wubu_hive_size(h) == 3, "size == 3 after 3 inserts");

    wubu_hive_destroy(h);
    printf("  test_insert: done\n");
}

/* Test 3: Block overflow — new block allocated */
void test_block_overflow(void) {
    wubu_hive_t *h = wubu_hive_create(2); /* small blocks for testing */
    int vals[5];
    for (int i = 0; i < 5; i++) vals[i] = i * 10;

    for (int i = 0; i < 5; i++) {
        check(wubu_hive_insert(h, &vals[i]) == 0, "insert value");
    }
    check(wubu_hive_size(h) == 5, "size == 5 after 5 inserts");
    check(wubu_hive_blocks(h) == 3, "3 blocks for 5 values (cap=2)");

    wubu_hive_destroy(h);
    printf("  test_block_overflow: done\n");
}

/* Test 4: Erase and slot reuse */
void test_erase_reuse(void) {
    wubu_hive_t *h = wubu_hive_create(4);
    int a = 1, b = 2, c = 3, d = 4;

    wubu_hive_insert(h, &a);
    wubu_hive_insert(h, &b);
    wubu_hive_insert(h, &c);
    wubu_hive_insert(h, &d);
    check(wubu_hive_size(h) == 4, "size == 4");

    /* Erase b (middle slot) */
    check(wubu_hive_erase(h, &b) == 0, "erase b");
    check(wubu_hive_size(h) == 3, "size == 3 after erase");
    check(wubu_hive_find(h, &b) == 0, "b not found after erase");
    check(wubu_hive_find(h, &a) == 1, "a still found");

    /* Insert e — should reuse b's slot */
    int e = 5;
    check(wubu_hive_insert(h, &e) == 0, "insert e (reuses b's slot)");
    check(wubu_hive_size(h) == 4, "size == 4 after reuse insert");
    check(wubu_hive_find(h, &e) == 1, "e found after insert");

    wubu_hive_destroy(h);
    printf("  test_erase_reuse: done\n");
}

/* Test 5: Find */
void test_find(void) {
    wubu_hive_t *h = wubu_hive_create(4);
    int x = 100, y = 200;

    check(wubu_hive_find(h, &x) == 0, "find absent value returns 0");
    wubu_hive_insert(h, &x);
    check(wubu_hive_find(h, &x) == 1, "find present value returns 1");
    check(wubu_hive_find(h, &y) == 0, "find different absent value returns 0");

    wubu_hive_destroy(h);
    printf("  test_find: done\n");
}

/* Test 6: Iterate */
static int hive_iter_count(void *value, size_t idx, void *ctx) {
    (void)value; (void)idx;
    int *count = (int *)ctx;
    (*count)++;
    return 0;
}

void test_iterate(void) {
    wubu_hive_t *h = wubu_hive_create(4);
    int vals[6] = {10, 20, 30, 40, 50, 60};
    int iter_count = 0;

    for (int i = 0; i < 6; i++)
        wubu_hive_insert(h, &vals[i]);

    /* Erase 3 values to create gaps */
    wubu_hive_erase(h, &vals[1]); /* 20 */
    wubu_hive_erase(h, &vals[3]); /* 40 */
    wubu_hive_erase(h, &vals[5]); /* 60 */

    /* Iterate should skip free slots */
    int count = 0;
    wubu_hive_iterate(h, hive_iter_count, &iter_count);
    check(iter_count == 3, "iterate sees exactly 3 live slots");

    wubu_hive_destroy(h);
    printf("  test_iterate: done\n");
}

/* Test 7: Size tracking */
void test_size_accuracy(void) {
    wubu_hive_t *h = wubu_hive_create(4);
    int a = 1, b = 2, c = 3;

    check(wubu_hive_size(h) == 0, "size 0 initially");
    wubu_hive_insert(h, &a);
    check(wubu_hive_size(h) == 1, "size 1 after 1 insert");
    wubu_hive_insert(h, &b);
    check(wubu_hive_size(h) == 2, "size 2 after 2 inserts");
    wubu_hive_erase(h, &a);
    check(wubu_hive_size(h) == 1, "size 1 after 1 erase");
    wubu_hive_erase(h, &b);
    check(wubu_hive_size(h) == 0, "size 0 after all erased");

    wubu_hive_destroy(h);
    printf("  test_size_accuracy: done\n");
}

/* Test 8: Pointer stability */
void test_pointer_stability(void) {
    wubu_hive_t *h = wubu_hive_create(2);
    int a = 1, b = 2;

    wubu_hive_insert(h, &a);
    wubu_hive_insert(h, &b);

    /* Erase a, insert c — a's slot is reused but c has a different pointer */
    int c = 3;
    wubu_hive_erase(h, &a);
    wubu_hive_insert(h, &c);

    /* b should still be findable (stable pointer) */
    check(wubu_hive_find(h, &b) == 1, "b still findable after a erased and c inserted");
    check(wubu_hive_find(h, &a) == 0, "a not found (was erased)");
    check(wubu_hive_find(h, &c) == 1, "c found");

    wubu_hive_destroy(h);
    printf("  test_pointer_stability: done\n");
}

int main(void) {
    printf("=== wubu_hive tests ===\n\n");

    test_create_destroy();
    test_insert();
    test_block_overflow();
    test_erase_reuse();
    test_find();
    test_iterate();
    test_size_accuracy();
    test_pointer_stability();

    printf("\n");
    if (errors == 0) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("%d TEST(S) FAILED\n", errors);
        return 1;
    }
}