/*
 * wubu_evict2026.c -- KV-eviction frontier mechanisms (Theme IO). C11.
 */
#include "wubu_evict2026.h"
#include <math.h>
#include <string.h>

int wubu_ev_pool_obs(const float *attn, int n, int w, float *out)
{
    if (!attn || !out || n <= 0 || w <= 0) return 0;
    int m = 0;
    for (int i = 0; i < n; i += w) {
        float mx = attn[i];
        for (int j = i; j < i + w && j < n; j++)
            if (attn[j] > mx) mx = attn[j];
        out[m++] = mx;
    }
    return m;
}

int wubu_ev_proxy_evict(const float *scores, int n, int keep, int *out_keep)
{
    if (!scores || !out_keep || n <= 0 || keep < 0) return -1;
    if (keep > n) keep = n;
    /* selection: top-keep indices by score (descending) */
    int idx[512];
    if (n > 512) n = 512;
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (scores[idx[j]] > scores[idx[i]]) {
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
    for (int i = 0; i < keep; i++) out_keep[i] = idx[i];
    return keep;
}

float wubu_ev_novelty(const float *proto, int n_proto, int dim,
                      const float *vec)
{
    if (!proto || !vec || n_proto <= 0 || dim <= 0) return 0;
    float best = -1;
    for (int i = 0; i < n_proto; i++) {
        float d2 = 0;
        for (int d = 0; d < dim; d++) {
            float dd = proto[i * dim + d] - vec[d];
            d2 += dd * dd;
        }
        if (best < 0 || d2 < best) best = d2;
    }
    return sqrtf(best);
}

uint32_t wubu_ev_simhash(const float *v, int dim, const float *plane, int seed)
{
    if (!v || !plane || dim <= 0) return 0;
    uint32_t h = 0;
    for (int d = 0; d < dim; d++) {
        float p = plane[d];
        if (seed & 1) p = -p;
        if (v[d] * p > 0) h |= (1u << (d & 31));
    }
    return h;
}

int wubu_ev_hamming(uint32_t a, uint32_t b)
{
    uint32_t x = a ^ b;
    int n = 0;
    while (x) { n += x & 1; x >>= 1; }
    return n;
}

int wubu_ev_twostage(const float *coarse, const float *query_sim,
                     int n, int coarse_keep, int final_keep, int *out)
{
    if (!coarse || !query_sim || !out || n <= 0) return -1;
    if (coarse_keep < 0 || final_keep < 0) return -1;
    if (coarse_keep > n) coarse_keep = n;
    int cand[512];
    if (n > 512) n = 512;
    /* stage 1: top-coarse_keep by coarse score */
    int ci[512];
    for (int i = 0; i < n; i++) ci[i] = i;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (coarse[ci[j]] > coarse[ci[i]]) {
                int t = ci[i]; ci[i] = ci[j]; ci[j] = t;
            }
    int n1 = coarse_keep;
    for (int i = 0; i < n1; i++) cand[i] = ci[i];
    /* stage 2: top-final_keep among candidates by query similarity */
    int k = final_keep < n1 ? final_keep : n1;
    for (int i = 0; i < n1; i++)
        for (int j = i + 1; j < n1; j++)
            if (query_sim[cand[j]] > query_sim[cand[i]]) {
                int t = cand[i]; cand[i] = cand[j]; cand[j] = t;
            }
    for (int i = 0; i < k; i++) out[i] = cand[i];
    return k;
}

int wubu_ev_adakv_budget(const float *dispersion, int n_heads,
                         int total_budget, int i)
{
    if (!dispersion || n_heads <= 0 || total_budget <= 0) return 0;
    if (i < 0 || i >= n_heads) return 0;
    float sum = 0;
    for (int h = 0; h < n_heads; h++) sum += dispersion[h];
    if (sum <= 0) return total_budget / n_heads;
    int b = (int)((float)total_budget * dispersion[i] / sum);
    return b < 0 ? 0 : b;
}

int wubu_ev_keysim_redundant(const float *k, const float *kept,
                             int dim, float thresh)
{
    if (!k || !kept || dim <= 0) return 0;
    float dot = 0, nk = 0, nq = 0;
    for (int d = 0; d < dim; d++) {
        dot += k[d] * kept[d];
        nk += k[d] * k[d];
        nq += kept[d] * kept[d];
    }
    if (nk <= 0 || nq <= 0) return 0;
    float cos = dot / (sqrtf(nk) * sqrtf(nq));
    return cos > thresh ? 1 : 0;
}

int wubu_ev_sink_pos(const float *attn, int n)
{
    if (!attn || n <= 0) return 0;
    int best = 0;
    for (int i = 1; i < n; i++)
        if (attn[i] > attn[best]) best = i;
    return best;
}

int wubu_ev_semantic_sponsor(float semantic, float thresh)
{
    return semantic >= thresh ? 1 : 0;
}

float wubu_ev_loss_bound(float dropped_mass, float total_mass)
{
    if (total_mass <= 0) return 0;
    float r = dropped_mass / total_mass;
    return r < 0 ? 0 : (r > 1 ? 1 : r);
}

float wubu_ev_block_drift(float drift_so_far, float step_drift, float cap)
{
    float d = drift_so_far + step_drift;
    if (cap > 0 && d > cap) d = cap;
    return d < 0 ? 0 : d;
}

int wubu_ev_reserve_sink(int budget, int sink_count, int outlier_count)
{
    if (budget <= 0) return 0;
    int r = sink_count + outlier_count;
    return r > budget ? budget : r;
}

int wubu_ev_hybrid_choose(float value, float evict_cost, float compress_cost)
{
    /* evict when the value is below the cost of keeping (compressing) */
    return value < compress_cost ? 1 : 0;
    (void)evict_cost;
}

float wubu_ev_stream_softmax(float *running_max, float *running_sum,
                             float logit)
{
    if (!running_max || !running_sum) return 0;
    if (logit > *running_max) {
        *running_sum *= expf(*running_max - logit);
        *running_max = logit;
    }
    float w = expf(logit - *running_max);
    *running_sum += w;
    return w / *running_sum;
}

int wubu_ev_hysteresis(float score, float thresh, float band, int state)
{
    if (band < 0) band = 0;
    if (state) return score > thresh - band ? 1 : 0;
    return score > thresh + band ? 1 : 0;
}

float wubu_ev_head_disparity(const float *dispersion, int n_heads)
{
    if (!dispersion || n_heads <= 0) return 1.0f;
    float mn = dispersion[0], mx = dispersion[0];
    for (int i = 1; i < n_heads; i++) {
        if (dispersion[i] < mn) mn = dispersion[i];
        if (dispersion[i] > mx) mx = dispersion[i];
    }
    if (mn <= 0) return mx > 0 ? 1e9f : 1.0f;
    return mx / mn;
}
