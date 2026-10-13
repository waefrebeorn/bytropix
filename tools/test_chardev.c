/* test_chardev.c -- KERNEL #25 character device driver framework.
 * Proves: register/open/read/write/release, per-handle cursor, EOF, append,
 * multiple independent opens, fail-closed. */
#include "wubu_chardev.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_chardev_t c;
    CHECK(wubu_cd_init(&c) == 0, "init ok");
    CHECK(wubu_cd_register(&c, "tty0", (const uint8_t*)"hello", 5) == 0, "register tty0 ('hello')");
    CHECK(wubu_cd_register(&c, "tty1", (const uint8_t*)"world", 5) == 0, "register tty1 ('world')");

    /* open + read all 5 bytes */
    int h = wubu_cd_open(&c, "tty0");
    CHECK(h >= 0, "open tty0 -> handle");
    uint8_t buf[16];
    int n = wubu_cd_read(&c, h, buf, 5);
    CHECK(n == 5, "read 5 bytes");
    buf[n] = 0;
    CHECK(strcmp((char*)buf, "hello") == 0, "read 'hello'");
    /* EOF */
    CHECK(wubu_cd_read(&c, h, buf, 5) == 0, "EOF -> 0 bytes");

    /* two independent opens have independent cursors */
    int h1 = wubu_cd_open(&c, "tty0");
    int h2 = wubu_cd_open(&c, "tty0");
    CHECK(h1 >= 0 && h2 >= 0, "two opens of tty0");
    uint8_t b1[8], b2[8];
    wubu_cd_read(&c, h1, b1, 2);   /* h1 reads "he" */
    wubu_cd_read(&c, h2, b2, 5);   /* h2 reads all "hello" (independent) */
    b1[2]=0; b2[5]=0;
    CHECK(strcmp((char*)b1, "he") == 0, "h1 read 'he'");
    CHECK(strcmp((char*)b2, "hello") == 0, "h2 independent, read 'hello'");

    /* write appends to the device buffer */
    int hw = wubu_cd_open(&c, "tty1");
    int wn = wubu_cd_write(&c, hw, (const uint8_t*)"!", 1);
    CHECK(wn == 1, "write 1 byte to tty1");
    /* re-read tty1 fresh: 'world!' */
    int h3 = wubu_cd_open(&c, "tty1");
    uint8_t b3[8];
    int rn = wubu_cd_read(&c, h3, b3, 8);
    CHECK(rn == 6, "tty1 now 6 bytes");
    b3[rn]=0;
    CHECK(strcmp((char*)b3, "world!") == 0, "tty1 reads 'world!'");

    /* release */
    CHECK(wubu_cd_release(&c, h) == 0, "release h");
    CHECK(wubu_cd_read(&c, h, buf, 5) == -1, "read after release -> -1");

    /* open unknown device */
    CHECK(wubu_cd_open(&c, "nope") == -1, "open unknown -> -1");

    /* fail-closed */
    CHECK(wubu_cd_read(&c, 999, buf, 5) == -1, "bad handle -> -1");
    CHECK(wubu_cd_write(&c, 999, (const uint8_t*)"x", 1) == -1, "write bad handle -> -1");
    CHECK(wubu_cd_release(&c, 999) == -1, "release bad handle -> -1");
    CHECK(wubu_cd_register(NULL, "x", NULL, 0) == -1, "NULL register -> -1");
    CHECK(wubu_cd_open(NULL, "x") == -1, "NULL open -> -1");

    printf("\n=== test_chardev: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
