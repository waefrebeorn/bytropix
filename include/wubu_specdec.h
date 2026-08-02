/*
 * wubu_specdec.h -- Speculative decoding: draft + verify + reject (HH01).
 */
#ifndef WUBU_SPECDEC_H
#define WUBU_SPECDEC_H

#define WUBU_SPECDEC_MAX_DRAFT 16
#define WUBU_SPECDEC_VOCAB 1024

/* Draft model: proposes tokens + gives draft probs. Target: gives target probs.
   We model both as arrays (prob over vocab) for a single position. */
typedef struct {
    int   draft_len;                         /* K tokens proposed */
    int   draft_tokens[WUBU_SPECDEC_MAX_DRAFT];
    float draft_probs[WUBU_SPECDEC_MAX_DRAFT][WUBU_SPECDEC_VOCAB];
    float target_probs[WUBU_SPECDEC_MAX_DRAFT][WUBU_SPECDEC_VOCAB];
    int   accepted[WUBU_SPECDEC_MAX_DRAFT];  /* 1 = accepted */
    int   n_accepted;
    int   bonus_token;                        /* target resample on full accept */
} wubu_specdec_t;

/* Run verification + rejection sampling. seed for stochastic accept/reject.
   Fills accepted[] + n_accepted + bonus_token. Returns total tokens produced
   (n_accepted + (bonus if all accepted)). */
int wubu_specdec_verify(wubu_specdec_t *sd, unsigned *seed);
/* Acceptance rate over last call. */
float wubu_specdec_rate(const wubu_specdec_t *sd);

#endif