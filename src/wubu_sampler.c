/* wubu_sampler.c -- BRAIN #84 sampler DRY anti-repetition for math.
 * See wubu_sampler.h. */
#include "wubu_sampler.h"
#include <math.h>

int wubu_sampler_init(wubu_sampler_t *s, int vocab_size,
                      float multiplier, float base, float allowed_length) {
    if (!s || vocab_size <= 0 || vocab_size > WUBU_SAMPLER_MAX_TOKENS) return -1;
    for (int i = 0; i < WUBU_SAMPLER_MAX_TOKENS; i++) s->counts[i] = 0;
    s->max_token = vocab_size;
    s->multiplier = multiplier > 0.0f ? multiplier : 1.0f;
    s->base = base > 1.0f ? base : 1.0f;
    s->allowed_length = allowed_length >= 0.0f ? allowed_length : 0.0f;
    return 0;
}

int wubu_sampler_commit(wubu_sampler_t *s, int token) {
    if (!s || token < 0 || token >= s->max_token) return -1;
    s->counts[token]++;
    return 0;
}

int wubu_sampler_apply_dry(wubu_sampler_t *s, float *logits, int n) {
    if (!s || !logits || n <= 0) return -1;
    int limit = n < s->max_token ? n : s->max_token;
    for (int t = 0; t < limit; t++) {
        int c = s->counts[t];
        if (c > (int)s->allowed_length) {
            /* penalty grows with excess repeats */
            float excess = (float)(c - (int)s->allowed_length);
            float penalty = s->multiplier * powf(s->base, excess);
            logits[t] /= penalty;
        }
    }
    return 0;
}

int wubu_sampler_argmax(const float *logits, int n) {
    if (!logits || n <= 0) return -1;
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}
