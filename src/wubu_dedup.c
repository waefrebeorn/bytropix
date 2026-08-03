/* wubu_dedup.c -- the rolling-hash duplicate-window scanner.
 * A polynomial rolling hash over the window; the hash table maps the
 * hash to the last-seen position; a window is a duplicate when the SAME
 * hash was seen AND the actual window contents match (the hash-collision
 * guard -- the hash alone is not the oracle). */
#include <stdlib.h>
#include <string.h>
#include "wubu_dedup.h"

#define HT_SIZE (1 << 20)
#define HT_MASK (HT_SIZE - 1)

typedef struct ent {
    long pos;
    uint64_t hash;
    struct ent *next;
} ent_t;

long wubu_dedup_scan(const uint16_t *toks, long n, int win, uint8_t *dup)
{
    if (!toks || n < win || win < 8) return 0;
    if (dup) memset(dup, 0, (size_t)n);
    ent_t **ht = (ent_t **)calloc(HT_SIZE, sizeof(ent_t *));
    if (!ht) return 0;
    ent_t *pool = (ent_t *)calloc((size_t)n + 1, sizeof(ent_t));
    if (!pool) { free(ht); return 0; }
    long ndup = 0, npool = 0;

    /* the initial hash + the rolling update */
    uint64_t h = 0, base = 1;
    for (int i = 0; i < win; i++) {
        h = h * 1315423911ull + toks[i] + 0x9e3779b97f4a7c15ull;
        base *= 1315423911ull;
    }
    for (long i = 0; i + win <= n; i++) {
        if (i > 0) {
            /* remove the leaving token, add the entering one */
            h = h * 1315423911ull + toks[i + win - 1] + 0x9e3779b97f4a7c15ull
                - base * (toks[i - 1] + 0x9e3779b97f4a7c15ull);
        }
        uint64_t idx = h & HT_MASK;
        int is_dup = 0;
        for (ent_t *e = ht[idx]; e; e = e->next) {
            if (e->hash != h) continue;
            /* the collision guard: compare the actual windows */
            long p = e->pos;
            int same = 1;
            for (int k = 0; k < win; k++)
                if (toks[p + k] != toks[i + k]) { same = 0; break; }
            if (same) { is_dup = 1; break; }
        }
        if (is_dup) {
            if (dup) dup[i] = 1;
            ndup++;
        } else {
            ent_t *e = &pool[npool++];
            e->pos = i;
            e->hash = h;
            e->next = ht[idx];
            ht[idx] = e;
        }
    }
    free(pool);
    free(ht);
    return ndup;
}

float wubu_dedup_rate(const uint8_t *dup, long n, int win)
{
    if (!dup || n < win) return 0;
    long d = 0, total = n - win + 1;
    for (long i = 0; i < total; i++) if (dup[i]) d++;
    return (float)d / (float)total;
}
