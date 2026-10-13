/* test_bonzi_mem.c -- UX #239 companion persistent memory.
 * Proves: set/get/forget, overwrite, eviction when full, lossless
 * serialize/restore round-trip, fail-closed on NULL/oversize. */
#include "wubu_bonzi_mem.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_bonzi_mem_t m;
    CHECK(wubu_bonzi_mem_init(&m) == 0, "init ok");

    /* set/get */
    CHECK(wubu_bonzi_mem_set(&m, "name", "WaefreBeorn") == 0, "set name");
    const char *v = wubu_bonzi_mem_get(&m, "name");
    CHECK(v && strcmp(v, "WaefreBeorn") == 0, "get name");
    CHECK(wubu_bonzi_mem_get(&m, "missing") == NULL, "get missing -> NULL");

    /* overwrite */
    wubu_bonzi_mem_set(&m, "name", "WB");
    CHECK(strcmp(wubu_bonzi_mem_get(&m, "name"), "WB") == 0, "overwrite");

    /* forget */
    CHECK(wubu_bonzi_mem_forget(&m, "name") == 0, "forget name");
    CHECK(wubu_bonzi_mem_get(&m, "name") == NULL, "gone after forget");
    CHECK(wubu_bonzi_mem_forget(&m, "name") == -1, "forget missing -> -1");

    /* count */
    wubu_bonzi_mem_set(&m, "a", "1");
    wubu_bonzi_mem_set(&m, "b", "2");
    CHECK(wubu_bonzi_mem_count(&m) == 2, "count 2");

    /* lossless serialize/restore round-trip */
    unsigned char buf[8192];
    int n = wubu_bonzi_mem_serialize(&m, buf, sizeof buf);
    CHECK(n > 0, "serialize ok");
    wubu_bonzi_mem_t m2;
    CHECK(wubu_bonzi_mem_restore(&m2, buf, (size_t)n) == 0, "restore ok");
    CHECK(wubu_bonzi_mem_count(&m2) == 2, "restored count 2");
    CHECK(strcmp(wubu_bonzi_mem_get(&m2, "a"), "1") == 0, "restored a=1");
    CHECK(strcmp(wubu_bonzi_mem_get(&m2, "b"), "2") == 0, "restored b=2");

    /* eviction when full */
    wubu_bonzi_mem_t full;
    wubu_bonzi_mem_init(&full);
    for (int i = 0; i < WUBU_BONZI_MEM_MAX + 10; i++) {
        char k[16]; snprintf(k, sizeof k, "k%d", i);
        wubu_bonzi_mem_set(&full, k, "v");
    }
    CHECK(wubu_bonzi_mem_count(&full) == WUBU_BONZI_MEM_MAX, "capped at MAX");
    CHECK(wubu_bonzi_mem_get(&full, "k0") == NULL, "oldest evicted");
    CHECK(wubu_bonzi_mem_get(&full, "k73") != NULL, "newest retained");

    /* fail-closed */
    CHECK(wubu_bonzi_mem_set(&m, NULL, "x") == -1, "NULL key -> -1");
    CHECK(wubu_bonzi_mem_set(&m, "k", NULL) == -1, "NULL val -> -1");
    CHECK(wubu_bonzi_mem_get(NULL, "k") == NULL, "NULL get -> NULL");
    CHECK(wubu_bonzi_mem_restore(&m, buf, 4) == -1, "truncated restore -> -1");
    CHECK(wubu_bonzi_mem_serialize(&m, buf, 4) == -1, "tiny buffer serialize -> -1");

    printf("\n=== test_bonzi_mem: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
