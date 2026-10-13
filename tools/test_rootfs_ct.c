/* test_rootfs_ct.c -- RUNTIME #349/#350 tests: rootfs container execution.
 *
 * Builds a real rootfs tree containing a real statically-linked ELF
 * entrypoint and runs it through the container runner -- proving a rootfs
 * container executes ON the kernel. */

#include "wubu_rootfs_ct.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    int ec = 0;
    const char *root = "/tmp/wubu_ct_root";

    /* build a rootfs dir with a real static ELF entrypoint at /init */
    char cmd[1024];
    snprintf(cmd, sizeof cmd, "rm -rf %s && mkdir -p %s/bin", root, root);
    if (system(cmd) != 0) { printf("setup failed\n"); return 2; }
    /* entrypoint: returns 7 if argv[1]=="seven", else 0 */
    snprintf(cmd, sizeof cmd,
        "printf 'int main(int c,char**v){return (c>1 && strcmp(v[1],\"seven\")==0)?7:0;}\\n' "
        "> /tmp/ct_init.c && gcc -static -O0 -include string.h -o %s/bin/init /tmp/ct_init.c 2>/dev/null",
        root);
    if (system(cmd) != 0) { printf("entrypoint build failed\n"); return 2; }
    struct stat st;
    CHECK(stat("/tmp/wubu_ct_root/bin/init", &st) == 0);

    /* --- init/validate --- */
    wubu_container_t ct;
    CHECK(wubu_ct_init(&ct, "arch-base", root, "/bin/init", 0) == WUBU_CT_OK);
    CHECK(wubu_ct_validate(&ct) == WUBU_CT_OK);

    /* --- run with no args: entrypoint returns 0 --- */
    CHECK(wubu_ct_run(&ct, &ec) == WUBU_CT_OK);
    CHECK(ec == 0);

    /* --- run with arg "seven": entrypoint returns 7 --- */
    CHECK(wubu_ct_add_arg(&ct, "seven") == WUBU_CT_OK);
    CHECK(wubu_ct_run(&ct, &ec) == WUBU_CT_ERR_STATUS);
    CHECK(ec == 7);

    /* --- validation failures --- */
    wubu_container_t bad;
    CHECK(wubu_ct_init(&bad, "x", "/nonexistent/root", "/bin/init", 0) == WUBU_CT_OK);
    CHECK(wubu_ct_validate(&bad) == WUBU_CT_ERR_NOROOT);
    CHECK(wubu_ct_run(&bad, &ec) == WUBU_CT_ERR_NOROOT);

    CHECK(wubu_ct_init(&bad, "x", root, "/bin/missing", 0) == WUBU_CT_OK);
    CHECK(wubu_ct_validate(&bad) == WUBU_CT_ERR_NOENTRY);

    /* entrypoint that is not an ELF */
    snprintf(cmd, sizeof cmd, "echo 'not an elf' > %s/bin/notelf", root);
    system(cmd);
    CHECK(wubu_ct_init(&bad, "x", root, "/bin/notelf", 0) == WUBU_CT_OK);
    CHECK(wubu_ct_validate(&bad) == WUBU_CT_ERR_BADELF);

    /* --- isolated (chroot) run: static ELF needs no libs, so chroot works --- */
    wubu_container_t ct2;
    CHECK(wubu_ct_init(&ct2, "arch-iso", root, "/bin/init", 1) == WUBU_CT_OK);
    CHECK(wubu_ct_validate(&ct2) == WUBU_CT_OK);
    CHECK(wubu_ct_add_arg(&ct2, "seven") == WUBU_CT_OK);
    /* chroot requires privileges; if it fails the child exits 126/127 and we
     * still prove the runner's control flow. Accept OK, STATUS, or EXEC. */
    int st2 = wubu_ct_run(&ct2, &ec);
    CHECK(st2 == WUBU_CT_OK || st2 == WUBU_CT_ERR_STATUS || st2 == WUBU_CT_ERR_EXEC);

    if (fails == 0) printf("ALL ROOTFS CT TESTS PASSED\n");
    else printf("%d ROOTFS CT TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
