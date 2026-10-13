/* test_synthfile.c -- NAMESPACE #195 9P synthetic files (read computes).
 * Proves: register synthetic file, read invokes the generator (computes
 * content), generator receives ctx, cap enforced, unknown file -> -1,
 * fail-closed on NULL. */
#include "wubu_synthfile.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

/* a generator that computes "count=<N>" from a counter in ctx */
static int gen_count(void *ctx, char *buf, size_t cap) {
    int *n = (int *)ctx;
    int len = snprintf(buf, cap, "count=%d", *n);
    return len;
}
/* a generator that returns a fixed string */
static int gen_static(void *ctx, char *buf, size_t cap) {
    (void)ctx;
    int len = snprintf(buf, cap, "hello");
    return len;
}

int main(void) {
    wubu_synthfs_t fs;
    CHECK(wubu_synthfs_init(&fs) == 0, "init ok");

    int counter = 42;
    CHECK(wubu_synthfs_add(&fs, "mem", gen_count, &counter) == 0, "add mem");
    CHECK(wubu_synthfs_add(&fs, "greeting", gen_static, NULL) == 0, "add greeting");

    /* read a synthetic file -> generator runs, computes content */
    char buf[64];
    int n = wubu_synthfs_read(&fs, "mem", buf, sizeof buf);
    CHECK(n == (int)strlen("count=42"), "mem read returns computed length");
    CHECK(strcmp(buf, "count=42") == 0, "mem read computes 'count=42'");

    /* the generator received ctx (counter) */
    CHECK(strstr(buf, "42") != NULL, "ctx threaded into generator");

    /* a static generator */
    n = wubu_synthfs_read(&fs, "greeting", buf, sizeof buf);
    CHECK(n == (int)strlen("hello"), "greeting read length");
    CHECK(strcmp(buf, "hello") == 0, "greeting read 'hello'");

    /* reading is recomputed each time (state change via ctx) */
    counter = 100;
    wubu_synthfs_read(&fs, "mem", buf, sizeof buf);
    CHECK(strcmp(buf, "count=100") == 0, "re-read recomputes (ctx changed)");

    /* cap is enforced: snprintf truncates safely into a small buffer (returns
     * the would-be length, but writes at most cap-1 bytes + NUL) */
    char small[6];
    n = wubu_synthfs_read(&fs, "mem", small, sizeof small);
    CHECK(strlen(small) < sizeof small && small[sizeof small - 1] == '\0',
          "read truncates safely within cap");

    /* unknown file -> -1 */
    CHECK(wubu_synthfs_read(&fs, "nonexistent", buf, sizeof buf) == -1, "unknown file -> -1");

    /* fail-closed on NULL / bad args */
    CHECK(wubu_synthfs_read(&fs, "mem", NULL, 64) == -1, "NULL buf -> -1");
    CHECK(wubu_synthfs_read(&fs, "mem", buf, 0) == -1, "zero cap -> -1");
    CHECK(wubu_synthfs_read(NULL, "mem", buf, 64) == -1, "NULL fs -> -1");
    CHECK(wubu_synthfs_add(&fs, "bad", NULL, NULL) == -1, "NULL gen -> -1");

    printf("\n=== test_synthfile: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
