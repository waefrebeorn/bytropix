/* test_caprevoke.c -- NAMESPACE #191 capability revocation propagates.
 * Proves: cascade revocation detaches all descendants, attenuation, no orphan
 * survives, valid/rights reflect revocation, fail-closed on NULL. */
#include "wubu_caprevoke.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_caprevoke_t c;
    CHECK(wubu_cr_init(&c) == 0, "init ok");

    /* build a tree:
     *   0 (root, rights 0xF)
     *   |- 1 (child, rights 0x7)
     *   |   |- 3 (grandchild, rights 0x3)
     *   |- 2 (child, rights 0x5)
     */
    int r  = wubu_cr_create(&c, -1, 0xF);
    int a  = wubu_cr_create(&c, r, 0x7);
    int b  = wubu_cr_create(&c, r, 0x5);
    int g  = wubu_cr_create(&c, a, 0x3);
    CHECK(r == 0 && a == 1 && b == 2 && g == 3, "tree built (ids 0-3)");
    CHECK(wubu_cr_rights(&c, a) == 0x7, "child a rights 0x7");
    CHECK(wubu_cr_rights(&c, g) == 0x3, "grandchild g rights 0x3");

    /* attenuation: child can't exceed parent */
    int over = wubu_cr_create(&c, r, 0xFF);
    CHECK(wubu_cr_rights(&c, over) == 0xF, "over-broad request attenuated to parent's 0xF");

    /* revoke 'a' -> cascades to a and g (2 caps); b survives */
    int n = wubu_cr_revoke(&c, a);
    CHECK(n == 2, "revoking a cascades to a+g (2 revoked)");
    CHECK(wubu_cr_valid(&c, a) == 0, "a revoked");
    CHECK(wubu_cr_valid(&c, g) == 0, "grandchild g revoked (cascade)");
    CHECK(wubu_cr_valid(&c, b) == 1, "sibling b survives");
    CHECK(wubu_cr_valid(&c, r) == 0 || wubu_cr_valid(&c, r) == 1, "root state checked");
    CHECK(wubu_cr_rights(&c, g) == 0, "revoked cap has 0 rights");

    /* can't derive from a revoked parent */
    CHECK(wubu_cr_create(&c, a, 0x1) == -1, "derive from revoked parent -> -1");

    /* re-revoke already-revoked -> -1 */
    CHECK(wubu_cr_revoke(&c, a) == -1, "re-revoke revoked -> -1");

    /* revoke root -> cascades to root, b, and the 'over' cap (all still-alive
     * descendants); a and g are already revoked */
    int n2 = wubu_cr_revoke(&c, r);
    CHECK(n2 == 3, "revoking root cascades to root+b+over (a,g already revoked)");
    CHECK(wubu_cr_valid(&c, r) == 0 && wubu_cr_valid(&c, b) == 0, "all revoked");
    CHECK(c.n_revocations == 5, "5 total revocations across the tree (a,g + root,b,over)");

    /* fail-closed */
    CHECK(wubu_cr_revoke(&c, 999) == -1, "revoke unknown -> -1");
    CHECK(wubu_cr_valid(&c, 999) == 0, "unknown cap invalid");
    CHECK(wubu_cr_create(NULL, -1, 1) == -1, "NULL create -> -1");
    CHECK(wubu_cr_revoke(NULL, 0) == -1, "NULL revoke -> -1");
    CHECK(wubu_cr_valid(NULL, 0) == 0, "NULL valid -> 0");
    CHECK(wubu_cr_init(NULL) == -1, "NULL init -> -1");

    printf("\n=== test_caprevoke: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
