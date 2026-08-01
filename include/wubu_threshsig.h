/*
 * wubu_threshsig.h -- Threshold signing (aggregate agent signatures) (DD02).
 */
#ifndef WUBU_THRESHSIG_H
#define WUBU_THRESHSIG_H

#define WUBU_THRESHSIG_MAX_SIGS 16

typedef struct {
    int  signer_id;
    unsigned sig;  /* simplified: deterministic pseudo-signature from signer_id + message_hash */
} wubu_ts_sig_t;

typedef struct {
    wubu_ts_sig_t sigs[WUBU_THRESHSIG_MAX_SIGS];
    int n_sigs;
    int threshold;  /* 2/3+1 of total nodes */
} wubu_threshsig_t;

/* Initialize with the BFT threshold (2/3+1 of n_nodes). */
int  wubu_threshsig_init(wubu_threshsig_t *ts, int n_nodes);
/* Sign: derive a pseudo-signature from signer_id + message hash. */
unsigned wubu_threshsig_sign(int signer_id, unsigned message_hash);
/* Add a signature. Returns 0 ok, -1 if full or already signed by this agent. */
int  wubu_threshsig_add(wubu_threshsig_t *ts, int signer_id, unsigned message_hash);
/* Check if threshold of unique signers reached. */
int  wubu_threshsig_verified(const wubu_threshsig_t *ts);

#endif