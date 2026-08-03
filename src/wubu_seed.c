/* wubu_seed.c -- the deterministic environment seed (splitmix64). */
#include "wubu_seed.h"

static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void wubu_seed_init(wubu_seed_t *r, uint64_t seed)
{
    r->state = seed ? seed : 1;   /* 0 is a degenerate splitmix state */
}

uint64_t wubu_seed_next(wubu_seed_t *r)
{
    return splitmix64(&r->state);
}

double wubu_seed_unit(wubu_seed_t *r)
{
    return (double)(wubu_seed_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

void wubu_seed_shuffle(wubu_seed_t *r, int *items, int n)
{
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(wubu_seed_unit(r) * (double)(i + 1));
        if (j > i) j = i;
        int t = items[i]; items[i] = items[j]; items[j] = t;
    }
}