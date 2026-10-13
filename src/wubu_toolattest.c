/* wubu_toolattest.c -- AGIEDR #309 supply-chain attestation of tools.
 * See wubu_toolattest.h. */
#include "wubu_toolattest.h"
#include <string.h>

int wubu_ta_init(wubu_toolattest_t *t) {
    if (!t) return -1;
    memset(t->entries, 0, sizeof t->entries);
    t->n = 0;
    return 0;
}

int wubu_ta_enroll(wubu_toolattest_t *t, const char *name,
                   unsigned long long hash, const char *signer) {
    if (!t || !name || !signer || t->n >= WUBU_TA_MAX) return -1;
    if (strlen(name) >= WUBU_TA_NAMELEN) return -1;
    if (strlen(signer) >= WUBU_TA_SIGNERLEN) return -1;
    wubu_ta_entry_t *e = &t->entries[t->n];
    strncpy(e->name, name, WUBU_TA_NAMELEN - 1); e->name[WUBU_TA_NAMELEN-1] = '\0';
    strncpy(e->signer, signer, WUBU_TA_SIGNERLEN - 1); e->signer[WUBU_TA_SIGNERLEN-1] = '\0';
    e->hash = hash;
    t->n++;
    return 0;
}

int wubu_ta_verify(const wubu_toolattest_t *t, const wubu_ta_attest_t *att) {
    if (!t || !att) return 0;   /* fail-closed */
    for (int i = 0; i < t->n; i++) {
        if (strcmp(t->entries[i].name, att->name) == 0) {
            /* must match BOTH the code hash and the trusted signer */
            if (t->entries[i].hash != att->hash) return 0;          /* tampered */
            if (strcmp(t->entries[i].signer, att->signer) != 0) return 0; /* untrusted signer */
            return 1;
        }
    }
    return 0;   /* unknown tool: not enrolled */
}
