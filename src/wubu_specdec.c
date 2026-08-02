/*
 * wubu_specdec.c -- Speculative decoding: draft + verify + reject (HH01). C11.
 *
 * Convergence (Leviathan/Chen speculative decoding + rejection sampling 7-hop):
 *   - HH01: draft model proposes K tokens; target verifies all in ONE forward
 *     pass. Accept token i with prob min(1, p_target/q_draft). On rejection,
 *     resample from residual norm(max(0, p-q)) and discard the rest. Output
 *     distribution is EXACTLY the target (no quality loss). At home: a small
 *     draft speculates next-K tokens for the 27-layer target; verification is
 *     1 target forward pass → speedup ∝ acceptance rate. Directly serves the
 *     27+ tok/s mandate.
 */
#include "wubu_specdec.h"
#include <math.h>
#include <string.h>

static double rng_f(unsigned *s) {
    *s = (*s * 1103515245U + 12345U) & 0x7fffffff;
    return (double)(*s) / (double)0x7fffffff;
}

int wubu_specdec_verify(wubu_specdec_t *sd, unsigned *seed) {
    if (!sd || !seed || sd->draft_len <= 0) return -1;
    int accepted = 0;
    int first_reject = -1;
    for (int i = 0; i < sd->draft_len; i++) {
        int x = sd->draft_tokens[i];
        float pt = sd->target_probs[i][x];
        float pq = sd->draft_probs[i][x];
        if (pq < 1e-8f) pq = 1e-8f;
        float ratio = pt / pq;
        float accept_p = (ratio > 1.0f) ? 1.0f : ratio;  /* min(1, p/q) */
        if (rng_f(seed) <= accept_p) {
            sd->accepted[i] = 1;
            accepted++;
        } else {
            sd->accepted[i] = 0;
            first_reject = i;
            break;  /* discard this + all subsequent drafts */
        }
    }
    sd->n_accepted = accepted;
    /* If all accepted → bonus token from target (sampled from target dist). */
    if (first_reject < 0) {
        /* sample from target_probs[draft_len-1] */
        double r = rng_f(seed);
        double cum = 0.0;
        int tok = WUBU_SPECDEC_VOCAB - 1;
        for (int v = 0; v < WUBU_SPECDEC_VOCAB; v++) {
            cum += sd->target_probs[sd->draft_len - 1][v];
            if (r <= cum) { tok = v; break; }
        }
        sd->bonus_token = tok;
        return accepted + 1;  /* K accepted + 1 bonus */
    } else {
        /* Resample from residual norm(max(0, p-q)) at rejection point. */
        int i = first_reject;
        float resid[WUBU_SPECDEC_VOCAB];
        double sum = 0.0;
        for (int v = 0; v < WUBU_SPECDEC_VOCAB; v++) {
            float r0 = sd->target_probs[i][v] - sd->draft_probs[i][v];
            resid[v] = (r0 > 0.0f) ? r0 : 0.0f;
            sum += resid[v];
        }
        if (sum < 1e-8) {
            /* fallback: sample target */
            double r = rng_f(seed); double cum = 0.0;
            int tok = 0;
            for (int v = 0; v < WUBU_SPECDEC_VOCAB; v++) {
                cum += sd->target_probs[i][v];
                if (r <= cum) { tok = v; break; }
            }
            sd->bonus_token = tok;
        } else {
            double r = rng_f(seed); double cum = 0.0; int tok = 0;
            for (int v = 0; v < WUBU_SPECDEC_VOCAB; v++) {
                cum += resid[v] / sum;
                if (r <= cum) { tok = v; break; }
            }
            sd->bonus_token = tok;
        }
        return accepted + 1;  /* accepted before reject + 1 resampled */
    }
}

float wubu_specdec_rate(const wubu_specdec_t *sd) {
    if (!sd || sd->draft_len <= 0) return 0.0f;
    return (float)sd->n_accepted / (float)sd->draft_len;
}
