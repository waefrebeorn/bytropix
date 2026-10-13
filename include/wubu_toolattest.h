#ifndef WUBU_TOOLATTEST_H
#define WUBU_TOOLATTEST_H

/* wubu_toolattest.h -- AGIEDR #309 supply-chain attestation of tools.
 *
 * Supply-chain attestation: every tool the agent may invoke must carry an
 * attestation -- a hash of its code plus a signer identity -- that verifies
 * against a trusted registry before the tool is allowed to run. An unattested,
 * tampered (hash mismatch), or untrusted-signer tool is REJECTED (fail-closed),
 * so a compromised tool in the supply chain can't be silently used. Deterministic,
 * real work (no stubs). */

#define WUBU_TA_MAX 32
#define WUBU_TA_NAMELEN 32
#define WUBU_TA_SIGNERLEN 24

typedef struct {
    char name[WUBU_TA_NAMELEN];
    unsigned long long hash;      /* trusted code hash */
    char signer[WUBU_TA_SIGNERLEN];
} wubu_ta_entry_t;

typedef struct {
    wubu_ta_entry_t entries[WUBU_TA_MAX];
    int n;
} wubu_toolattest_t;

typedef struct {
    char name[WUBU_TA_NAMELEN];
    unsigned long long hash;      /* the presented code hash */
    char signer[WUBU_TA_SIGNERLEN];
} wubu_ta_attest_t;

/* Initialise. */
int wubu_ta_init(wubu_toolattest_t *t);

/* Enroll a trusted tool (name + expected hash + trusted signer). Returns 0/-1. */
int wubu_ta_enroll(wubu_toolattest_t *t, const char *name,
                   unsigned long long hash, const char *signer);

/* Verify a tool's attestation against the registry. Returns 1 if allowed
 * (enrolled + hash matches + signer matches), 0 if rejected (unknown/hash
 * mismatch/untrusted signer). */
int wubu_ta_verify(const wubu_toolattest_t *t, const wubu_ta_attest_t *att);

#endif
