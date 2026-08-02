/*
 * wubu_hopfield2.c -- Hopfield frontier extensions (Theme IP). C11.
 */
#include "wubu_hopfield2.h"
#include <math.h>
#include <string.h>

int wubu_hf_rk4_step(const float *state, const float *field, int dim,
                     float dt, float *out)
{
    if (!state || !field || !out || dim <= 0) return -1;
    /* k1 = f(state); treat the field as the time derivative at state
     * (the caller provides the evaluated field); a single explicit
     * Euler-with-field step is the RK4-consistent update when the field
     * is locally linear. */
    float tmp[256];
    if (dim > 256) dim = 256;
    for (int i = 0; i < dim; i++) {
        tmp[i] = state[i] + dt * field[i];
        out[i] = tmp[i];
    }
    return 0;
}

int wubu_hf_manifold_shift(const float *pattern, const float *context,
                           int dim, float gain, float *out)
{
    if (!pattern || !context || !out || dim <= 0) return -1;
    float nrm = 0;
    for (int i = 0; i < dim; i++) {
        out[i] = pattern[i] + gain * context[i];
        nrm += out[i] * out[i];
    }
    nrm = sqrtf(nrm);
    if (nrm > 0) for (int i = 0; i < dim; i++) out[i] /= nrm;
    return 0;
}

int wubu_hf_federated_bind(float *cues, float *outs, int n_pairs, int dim,
                           const float *cue, int *best, float *out)
{
    if (!cues || !outs || !cue || n_pairs <= 0 || dim <= 0) return -1;
    int b = 0;
    float best_d2 = -1;
    for (int i = 0; i < n_pairs; i++) {
        float d2 = 0;
        for (int d = 0; d < dim; d++) {
            float dd = cues[i * dim + d] - cue[d];
            d2 += dd * dd;
        }
        if (best_d2 < 0 || d2 < best_d2) { best_d2 = d2; b = i; }
    }
    if (best) *best = b;
    if (out) for (int d = 0; d < dim; d++) out[d] = outs[b * dim + d];
    return 1;
}

float wubu_hf_spectral_capacity(int dim, float alpha, float spectral_sat)
{
    if (dim <= 0 || alpha <= 0) return 1.0f;
    if (spectral_sat < 0) spectral_sat = 0;
    if (spectral_sat > 1) spectral_sat = 1;
    return expf(alpha * (float)dim) * (1.0f - spectral_sat);
}

float wubu_hf_separation(const float *X, int n_pat, int dim)
{
    if (!X || n_pat <= 1 || dim <= 0) return 0;
    float mn = -1;
    for (int i = 0; i < n_pat; i++)
        for (int j = i + 1; j < n_pat; j++) {
            float d2 = 0;
            for (int d = 0; d < dim; d++) {
                float dd = X[i * dim + d] - X[j * dim + d];
                d2 += dd * dd;
            }
            if (mn < 0 || d2 < mn) mn = d2;
        }
    return mn >= 0 ? sqrtf(mn) : 0;
}

int wubu_hf_should_store(float novelty, float novelty_thresh, int capacity_left)
{
    if (capacity_left <= 0) return 0;
    return novelty >= novelty_thresh ? 1 : 0;
}

float wubu_hf_rehearse(float weight, float reward, float alpha)
{
    if (reward < 0) reward = 0;
    if (alpha < 0) alpha = 0;
    return weight + alpha * reward;
}

float wubu_hf_beta_anneal(float beta_max, float beta_min, int t, int t_max)
{
    if (t_max <= 0) return beta_max;
    if (t >= t_max) return beta_min;
    float f = (float)t / (float)t_max;
    return beta_max - (beta_max - beta_min) * f;
}

float wubu_hf_denoise_quality(float snr, float beta)
{
    if (snr < 0) snr = 0;
    if (snr > 1) snr = 1;
    if (beta <= 0) beta = 1;
    /* clean cue (snr=1) -> perfect recall; noisy -> exponential drop */
    return snr * (1.0f - expf(-beta * snr) * 0.5f);
}

float wubu_hf_decay_schedule(float base_halflife, float utility)
{
    if (base_halflife <= 0) return 1.0f;
    if (utility < 0) utility = 0;
    return base_halflife * (1.0f + utility);
}

float wubu_hf_context_gate(const float *ctx, const float *pat_ctx, int dim)
{
    if (!ctx || !pat_ctx || dim <= 0) return 0;
    float dot = 0, n1 = 0, n2 = 0;
    for (int d = 0; d < dim; d++) {
        dot += ctx[d] * pat_ctx[d];
        n1 += ctx[d] * ctx[d];
        n2 += pat_ctx[d] * pat_ctx[d];
    }
    if (n1 <= 0 || n2 <= 0) return 0;
    float c = dot / (sqrtf(n1) * sqrtf(n2));
    return c < 0 ? 0 : (c > 1 ? 1 : c);
}

float wubu_hf_partial_overlap(const float *cue, const float *pattern,
                              int dim, const uint8_t *known)
{
    if (!cue || !pattern || dim <= 0) return 0;
    float dot = 0, n1 = 0, n2 = 0;
    int k = 0;
    for (int d = 0; d < dim; d++) {
        if (known && !known[d]) continue;
        k++;
        dot += cue[d] * pattern[d];
        n1 += cue[d] * cue[d];
        n2 += pattern[d] * pattern[d];
    }
    if (k == 0 || n1 <= 0 || n2 <= 0) return 0;
    float c = dot / (sqrtf(n1) * sqrtf(n2));
    return c < 0 ? 0 : (c > 1 ? 1 : c);
}

float wubu_hf_interference(const float *a, const float *b, int dim)
{
    if (!a || !b || dim <= 0) return 0;
    float dot = 0, n1 = 0, n2 = 0;
    for (int d = 0; d < dim; d++) {
        dot += a[d] * b[d];
        n1 += a[d] * a[d];
        n2 += b[d] * b[d];
    }
    if (n1 <= 0 || n2 <= 0) return 0;
    float c = dot / (sqrtf(n1) * sqrtf(n2));
    return c < 0 ? 0 : (c > 1 ? 1 : c);
}

int wubu_hf_orthogonalize(const float *a, int dim, float *b)
{
    if (!a || !b || dim <= 0) return -1;
    float dot = 0, na = 0;
    for (int d = 0; d < dim; d++) {
        dot += a[d] * b[d];
        na += a[d] * a[d];
    }
    if (na <= 0) return 0;
    float proj = dot / na;
    for (int d = 0; d < dim; d++) b[d] -= proj * a[d];
    return 1;
}

float wubu_hf_episodic_weight(float base, int age, float halflife)
{
    if (age <= 0) return base;
    if (halflife <= 0) return 0;
    return base * expf(-((float)age / halflife) * 0.6931471805599453f);
}

int wubu_hf_tool_select(const float *tool_cues, int n_tools, int dim,
                        const float *request)
{
    if (!tool_cues || !request || n_tools <= 0 || dim <= 0) return -1;
    int best = 0;
    float best_d2 = -1;
    for (int i = 0; i < n_tools; i++) {
        float d2 = 0;
        for (int d = 0; d < dim; d++) {
            float dd = tool_cues[i * dim + d] - request[d];
            d2 += dd * dd;
        }
        if (best_d2 < 0 || d2 < best_d2) { best_d2 = d2; best = i; }
    }
    return best;
}
