/* test_toolattest.c -- AGIEDR #309 supply-chain attestation of tools.
 * Proves: enrolled+matching tool allowed, tampered (hash mismatch) rejected,
 * untrusted signer rejected, unknown tool rejected, fail-closed on NULL. */
#include "wubu_toolattest.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_toolattest_t t;
    CHECK(wubu_ta_init(&t) == 0, "init ok");
    CHECK(wubu_ta_enroll(&t, "grep", 0xAAAAULL, "wubutech") == 0, "enroll grep");
    CHECK(wubu_ta_enroll(&t, "calc", 0xBBBBULL, "wubutech") == 0, "enroll calc");

    /* valid attestation allowed */
    wubu_ta_attest_t good = {"grep", 0xAAAAULL, "wubutech"};
    CHECK(wubu_ta_verify(&t, &good) == 1, "valid grep attestation allowed");

    /* tampered code (hash mismatch) rejected */
    wubu_ta_attest_t tampered = {"grep", 0xDEADULL, "wubutech"};
    CHECK(wubu_ta_verify(&t, &tampered) == 0, "tampered grep (hash mismatch) rejected");

    /* untrusted signer rejected (hash matches but signer wrong) */
    wubu_ta_attest_t badsigner = {"grep", 0xAAAAULL, "evilcorp"};
    CHECK(wubu_ta_verify(&t, &badsigner) == 0, "untrusted signer rejected");

    /* unknown (unenrolled) tool rejected */
    wubu_ta_attest_t unknown = {"backdoor", 0x1234ULL, "wubutech"};
    CHECK(wubu_ta_verify(&t, &unknown) == 0, "unknown tool rejected");

    /* calc valid */
    wubu_ta_attest_t cgood = {"calc", 0xBBBBULL, "wubutech"};
    CHECK(wubu_ta_verify(&t, &cgood) == 1, "valid calc attestation allowed");

    /* fail-closed on NULL */
    CHECK(wubu_ta_init(NULL) == -1, "NULL init -> -1");
    CHECK(wubu_ta_enroll(NULL, "x", 1, "y") == -1, "NULL enroll -> -1");
    CHECK(wubu_ta_verify(&t, NULL) == 0, "NULL attest -> 0 (fail-closed)");
    CHECK(wubu_ta_verify(NULL, &good) == 0, "NULL registry -> 0 (fail-closed)");

    printf("\n=== test_toolattest: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
