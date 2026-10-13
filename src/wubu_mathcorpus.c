/* wubu_mathcorpus.c -- BRAIN #124 math corpus streamed to training.
 * See wubu_mathcorpus.h. */
#include "wubu_mathcorpus.h"
#include <string.h>

static uint32_t lcg(uint32_t *s) { *s = (*s * 1103515245U + 12345U) & 0x7fffffff; return *s; }

int wubu_mc_init(wubu_mathcorpus_t *c, int n_examples, uint32_t seed) {
    if (!c || n_examples <= 0 || n_examples > WUBU_MC_MAX_EXAMPLES) return -1;
    memset(c, 0, sizeof *c);
    c->n_examples = n_examples;
    c->seed = seed;
    c->pos = 0;
    /* deterministic seeded Fisher-Yates over the index permutation */
    for (int i = 0; i < n_examples; i++) c->order[i] = i;
    uint32_t s = seed ? seed : 1;
    for (int i = n_examples - 1; i > 0; i--) {
        int j = (int)(lcg(&s) % (uint32_t)(i + 1));
        int t = c->order[i]; c->order[i] = c->order[j]; c->order[j] = t;
    }
    c->initialized = 1;
    return 0;
}

int wubu_mc_synth(const wubu_mathcorpus_t *c, int index, wubu_mc_example_t *out) {
    if (!c || !out || index < 0 || index >= c->n_examples) return -1;
    /* deterministic content derived purely from the index (constant memory) */
    int len = 4 + (index % (WUBU_MC_TOKLEN - 4));
    for (int i = 0; i < len; i++)
        out->tokens[i] = (index * 31 + i * 7 + 1) % 50000;
    out->len = len;
    out->label = (index % 3 == 0) ? 0 : 1;   /* ~2/3 valid proof steps */
    return 0;
}

int wubu_mc_next(wubu_mathcorpus_t *c, wubu_mc_example_t *out) {
    if (!c || !c->initialized || !out) return -1;
    if (c->pos >= c->n_examples) return 0;   /* end of epoch */
    int logical = c->order[c->pos];
    if (wubu_mc_synth(c, logical, out) != 0) return -1;
    c->pos++;
    return 1;
}

int wubu_mc_rewind(wubu_mathcorpus_t *c, uint32_t reseed) {
    if (!c) return -1;
    if (reseed != 0) return wubu_mc_init(c, c->n_examples, reseed);
    c->pos = 0;   /* same shuffle, restart */
    return 0;
}

int wubu_mc_position(const wubu_mathcorpus_t *c) {
    return c ? c->pos : -1;
}
