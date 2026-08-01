/*
 * test_capzero.c -- AF02 (deny-by-default), AF03 (mem crypt), AF04 (NHI) verification.
 */
#include "wubu_capzero.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_capzero (AF02-AF04) ===\n");

    /* AF02 deny-by-default tool registry */
    wubu_capset_t *c = wubu_capset_create();
    CHECK(c != NULL, "capset created");
    CHECK(wubu_cap_check(c, "fs.read") == 0, "empty set -> deny (default)");
    wubu_cap_grant(c, "fs.read");
    wubu_cap_grant(c, "index.upsert");
    CHECK(wubu_cap_check(c, "fs.read") == 1, "granted tool allowed");
    CHECK(wubu_cap_check(c, "index.upsert") == 1, "second granted allowed");
    CHECK(wubu_cap_check(c, "fs.write") == 0, "unlisted tool denied");
    int dup = wubu_cap_grant(c, "fs.read"); /* duplicate grant */
    CHECK(dup == 1, "duplicate grant idempotent (returns 1)");
    wubu_capset_destroy(c);

    /* AF04 non-human identity */
    uint64_t t1 = wubu_nhi_issue("agent_7", "s3cr3t");
    uint64_t t2 = wubu_nhi_issue("agent_7", "s3cr3t");
    uint64_t t3 = wubu_nhi_issue("agent_8", "s3cr3t");
    CHECK(t1 == t2, "same id+secret -> stable token");
    CHECK(t1 != t3, "different id -> different token");
    CHECK(wubu_nhi_valid(t1) == 1, "issued token valid");
    CHECK(wubu_nhi_valid(0ULL) == 0, "zero token invalid");

    /* AF03 encrypted memory at rest: encrypt then decrypt restores original */
    unsigned char blob[24];
    for (int i = 0; i < 24; i++) blob[i] = (unsigned char)(i * 7 + 3);
    unsigned char orig[24];
    memcpy(orig, blob, 24);
    uint64_t key = t1, nonce = 0xABCDEFULL;
    wubu_mem_crypt(key, nonce, blob, 24);
    int changed = 0;
    for (int i = 0; i < 24; i++) if (blob[i] != orig[i]) changed = 1;
    CHECK(changed == 1, "ciphertext differs from plaintext");
    wubu_mem_crypt(key, nonce, blob, 24); /* decrypt (symmetric) */
    int restored = 1;
    for (int i = 0; i < 24; i++) if (blob[i] != orig[i]) restored = 0;
    CHECK(restored == 1, "decrypt restores original blob");
    /* wrong key -> does not restore */
    unsigned char blob2[24];
    memcpy(blob2, orig, 24);
    wubu_mem_crypt(key, nonce, blob2, 24);
    wubu_mem_crypt(key ^ 0xFFULL, nonce, blob2, 24); /* wrong key decrypt */
    int mismatch = 0;
    for (int i = 0; i < 24; i++) if (blob2[i] != orig[i]) mismatch = 1;
    CHECK(mismatch == 1, "wrong key fails to restore (confidentiality)");

    if (failures == 0) { printf("ALL CAPZERO TESTS PASSED\n"); return 0; }
    printf("%d CAPZERO TEST(S) FAILED\n", failures);
    return 1;
}
