/* test_abi_shim.c -- RUNTIME #339 tests: foreign-ABI translation. */
#include "wubu_abi_shim.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_abi_shim_t s;
    int nid;
    int64_t nargs[WUBU_ABI_MAX_ARGS];

    CHECK(wubu_abi_init(&s) == 0);
    CHECK(wubu_abi_translated(&s) == 0);
    CHECK(wubu_abi_denied(&s) == 0);

    /* Linux write(1) = fd, buf, count -> VSL WRITE, 3 args, pass-through */
    int64_t w1[] = {1, 0x1000, 12};
    CHECK(wubu_abi_translate(&s, 1 /*LNX_WRITE*/, w1, 3, &nid, nargs) == 3);
    CHECK(nid == WUBU_VSL_WRITE);
    CHECK(nargs[0] == 1 && nargs[1] == 0x1000 && nargs[2] == 12);

    /* Linux read(0) -> VSL READ */
    int64_t r1[] = {0, 0x2000, 64};
    CHECK(wubu_abi_translate(&s, 0 /*LNX_READ*/, r1, 3, &nid, nargs) == 3);
    CHECK(nid == WUBU_VSL_READ);
    CHECK(nargs[0] == 0 && nargs[1] == 0x2000 && nargs[2] == 64);

    /* Linux exit(42) -> VSL EXIT, 1 arg */
    int64_t e1[] = {42};
    CHECK(wubu_abi_translate(&s, 60 /*LNX_EXIT*/, e1, 1, &nid, nargs) == 1);
    CHECK(nid == WUBU_VSL_EXIT);
    CHECK(nargs[0] == 42);

    /* mmap: 6 args */
    int64_t m1[] = {0, 4096, 3, 0x22, -1, 0};
    CHECK(wubu_abi_translate(&s, 9 /*LNX_MMAP*/, m1, 6, &nid, nargs) == 6);
    CHECK(nid == WUBU_VSL_MMAP);
    CHECK(nargs[0]==0 && nargs[1]==4096 && nargs[2]==3 && nargs[3]==0x22 &&
          nargs[4]==-1 && nargs[5]==0);

    /* unknown foreign syscall -> denied (fail-closed) */
    CHECK(wubu_abi_translate(&s, 9999, w1, 3, &nid, nargs) == -1);

    /* explicitly map a foreign nr to NOSYS -> denied */
    CHECK(wubu_abi_map(&s, 1234, WUBU_VSL_NOSYS, 0, NULL) == 0);
    CHECK(wubu_abi_translate(&s, 1234, w1, 3, &nid, nargs) == -1);

    /* counters */
    CHECK(wubu_abi_translated(&s) == 4);   /* write, read, exit, mmap */
    CHECK(wubu_abi_denied(&s) == 2);       /* 9999, 1234 */

    /* override an existing mapping: remap open(2) with a reordered arg_regs */
    int reorder[3] = {2, 0, 1};   /* native slot0<-foreign2, slot1<-foreign0, slot2<-foreign1 */
    CHECK(wubu_abi_map(&s, 2 /*LNX_OPEN*/, WUBU_VSL_OPEN, 3, reorder) == 0);
    int64_t o1[] = {100, 200, 300};   /* foreign args */
    CHECK(wubu_abi_translate(&s, 2, o1, 3, &nid, nargs) == 3);
    CHECK(nargs[0] == 300 && nargs[1] == 100 && nargs[2] == 200);  /* reordered */

    /* fewer foreign args than native slots -> missing args become 0 */
    CHECK(wubu_abi_map(&s, 55, WUBU_VSL_READ, 3, NULL) == 0);  /* identity slots */
    int64_t few[] = {7};
    CHECK(wubu_abi_translate(&s, 55, few, 1, &nid, nargs) == 3);
    CHECK(nargs[0] == 7 && nargs[1] == 0 && nargs[2] == 0);

    /* default_deny toggle doesn't change unknown-syscall behaviour (still
     * fail-closed: unmapped has no safe target), but the setter works */
    CHECK(wubu_abi_set_default_deny(&s, 0) == 0);
    CHECK(wubu_abi_translate(&s, 8888, w1, 3, &nid, nargs) == -1);
    CHECK(wubu_abi_set_default_deny(&s, 1) == 0);

    /* bad args */
    CHECK(wubu_abi_translate(&s, 1, NULL, 0, &nid, nargs) == 3);  /* args missing -> 0s */
    CHECK(nargs[0] == 0);
    CHECK(wubu_abi_translate(NULL, 1, w1, 3, &nid, nargs) == -1);
    CHECK(wubu_abi_map(&s, 1, WUBU_VSL_WRITE, 99, NULL) == -1);   /* n_args too big */
    CHECK(wubu_abi_init(NULL) == -1);

    if (fails == 0) printf("ALL ABI_SHIM TESTS PASSED\n");
    else printf("%d ABI_SHIM TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
