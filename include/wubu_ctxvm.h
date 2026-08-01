/*
 * wubu_ctxvm.h -- AGI-OS context virtual-memory hierarchy (AF08-AF10).
 */
#ifndef WUBU_CTXVM_H
#define WUBU_CTXVM_H

#define WUBU_CTX_L1 1  /* gen window */
#define WUBU_CTX_L2 2  /* session */
#define WUBU_CTX_L3 3  /* long-term */
#define WUBU_CTX_L4 4  /* cross-session */

typedef struct {
    long *tok;
    int   head;
    int   size;
    int   capacity;
} wubu_ctxring_t;

int  wubu_ctx_tier(float importance, long ttl);                 /* AF08 */
int  wubu_ctx_evict_fifo(wubu_ctxring_t *r, long tok);          /* AF09 */
int  wubu_ctx_resident(const wubu_ctxring_t *r, long tok, int ws); /* AF09 */
float wubu_cosine(const float *a, const float *b, int n);      /* AF10 */
int  wubu_sem_cache_hit(const float *q, const float *c, int n, float thr); /* AF10 */

#endif
