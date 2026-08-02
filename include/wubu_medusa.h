/*
 * wubu_medusa.h -- Medusa self-draft heads (parallel tree draft) (HH05).
 */
#ifndef WUBU_MEDUSA_H
#define WUBU_MEDUSA_H

#define WUBU_MEDUSA_HEADS 4
#define WUBU_MEDUSA_BRANCH 2   /* tokens per head (tree breadth) */
#define WUBU_MEDUSA_VOCAB 32000

typedef struct {
    int n_heads;
    int branch;
    /* Each head h proposes `branch` candidate tokens (parallel). */
    int   candidates[WUBU_MEDUSA_HEADS][WUBU_MEDUSA_BRANCH];
    float probs[WUBU_MEDUSA_HEADS][WUBU_MEDUSA_BRANCH];
    /* Empirical acceptance from history (EMA) → adaptive draft length. */
    float accept_ema[WUBU_MEDUSA_HEADS];
    int   draft_len;   /* current effective draft length (heads used) */
} wubu_medusa_t;

/* Init with default acceptance EMA. */
int  wubu_medusa_init(wubu_medusa_t *m);
/* Adapt draft length: drop heads whose EMA acceptance < threshold. */
int  wubu_medusa_adapt(wubu_medusa_t *m, float threshold);
/* Update EMA after a verification round (n_accepted out of n_proposed). */
int  wubu_medusa_update(wubu_medusa_t *m, int head, int accepted, int proposed);

#endif