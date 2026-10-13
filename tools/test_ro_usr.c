/* test_ro_usr.c -- OMARCHY #289 immutable /usr read-only root.
 * Proves: /usr etc read-only, /var etc writable, write to read-only denied
 * (fail-closed), boundary match (/usr doesn't match /username), unknown -> RO,
 * fail-closed on NULL. */
#include "wubu_ro_usr.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_ro_usr_t p;
    CHECK(wubu_ro_init(&p) == 0, "init ok");

    /* system dirs are read-only */
    CHECK(wubu_ro_classify(&p, "/usr") == WUBU_FS_READONLY, "/usr read-only");
    CHECK(wubu_ro_classify(&p, "/usr/bin/gcc") == WUBU_FS_READONLY, "/usr/bin/gcc read-only");
    CHECK(wubu_ro_classify(&p, "/etc/passwd") == WUBU_FS_READONLY, "/etc/passwd read-only");
    CHECK(wubu_ro_classify(&p, "/boot/vmlinuz") == WUBU_FS_READONLY, "/boot read-only");
    CHECK(wubu_ro_classify(&p, "/lib/x.so") == WUBU_FS_READONLY, "/lib read-only");

    /* state dirs are writable */
    CHECK(wubu_ro_classify(&p, "/var/log/x") == WUBU_FS_WRITABLE, "/var writable");
    CHECK(wubu_ro_classify(&p, "/tmp/x") == WUBU_FS_WRITABLE, "/tmp writable");
    CHECK(wubu_ro_classify(&p, "/home/u") == WUBU_FS_WRITABLE, "/home writable");

    /* write to read-only denied */
    CHECK(wubu_ro_may_write(&p, "/usr/bin/gcc") == 0, "write /usr denied");
    CHECK(wubu_ro_may_write(&p, "/etc/passwd") == 0, "write /etc denied");
    /* write to writable allowed */
    CHECK(wubu_ro_may_write(&p, "/var/log/x") == 1, "write /var allowed");
    CHECK(wubu_ro_may_write(&p, "/tmp/x") == 1, "write /tmp allowed");

    /* boundary: /usr must NOT match a sibling like /usrx (not /usr/) */
    CHECK(wubu_ro_may_write(&p, "/usrx/data") == 0, "/usrx unknown -> read-only (not /usr)");
    /* a writable path that shares a prefix char-set boundary */
    CHECK(wubu_ro_may_write(&p, "/var") == 1, "/var itself writable");

    /* unknown path -> read-only (fail-closed) */
    CHECK(wubu_ro_classify(&p, "/mystery") == WUBU_FS_READONLY, "unknown path -> read-only");
    CHECK(wubu_ro_may_write(&p, "/mystery/x") == 0, "write unknown denied (fail-closed)");

    /* fail-closed on NULL */
    CHECK(wubu_ro_classify(NULL, "/var") == WUBU_FS_READONLY, "NULL classify -> read-only");
    CHECK(wubu_ro_may_write(NULL, "/var") == 0, "NULL may_write -> 0 (fail-closed)");
    CHECK(wubu_ro_init(NULL) == -1, "NULL init -> -1");

    printf("\n=== test_ro_usr: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
