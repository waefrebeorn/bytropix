/*
 * wubu_evict2026c.c -- the KV-eviction frontier, final (IO). C11.
 */
#include "wubu_evict2026c.h"
#include <math.h>
#include <string.h>

int wubu_evictc_h2o(const float *attention, int n, float th, int *keep)
{
    if (!attention || !keep || n <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n; i++)
        if (attention[i] >= th) keep[k++] = i;
    return k;
}

int wubu_evictc_sink(int *tokens, int n, int sink, int keep_n)
{
    if (!tokens || n <= 0 || sink < 0 || keep_n <= 0) return -1;
    int keep = n < keep_n ? n : keep_n;
    /* keep the sink tokens + the last (keep - sink) tokens */
    int keep_tail = keep - (sink < keep ? sink : keep);
    if (keep_tail < 0) keep_tail = 0;
    /* the result: sink count + tail count */
    (void)tokens;
    return sink + keep_tail;
}

int wubu_evictc_kvquant(const float *kv, int n, int32_t *quant, int32_t *outlier)
{
    if (!kv || !quant || !outlier) return -1;
    int32_t scale = 7;  /* 3-bit: -4..3 */
    for (int i = 0; i < n; i++) {
        float v = kv[i] < -1 ? -1 : (kv[i] > 1 ? 1 : kv[i]);
        quant[i] = (int32_t)(v * scale);
        if (fabsf(kv[i]) > 0.9f) outlier[i] = (int32_t)(kv[i] * 1000);
    }
    return n;
}

int wubu_evictc_track(float *running_sum, int i, float new_val)
{
    if (!running_sum || i < 0) return -1;
    running_sum[i] += new_val;
    return 0;
}

int wubu_evictc_recon_importance(const float *orig, const float *recon, int n, float th)
{
    if (!orig || !recon || n <= 0) return -1;
    float err = 0;
    for (int i = 0; i < n; i++) {
        float d = orig[i] - recon[i];
        err += d * d;
    }
    return (sqrtf(err / (float)n) < th) ? 1 : 0;
}

int wubu_evictc_outlier(const float *kv, int n, float th, int *sparse_idx)
{
    if (!kv || !sparse_idx || n <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n; i++)
        if (fabsf(kv[i]) > th) sparse_idx[k++] = i;
    return k;
}

int wubu_evictc_page_import(const float *pages, int n_pages, float th)
{
    if (!pages || n_pages <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n_pages; i++)
        if (pages[i] >= th) k++;
    return k;
}

float wubu_evictc_lsh_thresh(float correlation, float base)
{
    return base * (1.0f - correlation);
}

int wubu_evictc_proxy(int prompt_len, int *proxy_count)
{
    if (!proxy_count) return -1;
    *proxy_count = (prompt_len + 99) / 100;
    return 0;
}

int wubu_evictc_rope_reencode(const int *positions, int n, int shift, int *new_pos)
{
    if (!positions || !new_pos || n <= 0) return -1;
    for (int i = 0; i < n; i++) new_pos[i] = positions[i] - shift;
    return n;
}

int wubu_evictc_audit(float perplexity, float th)
{
    return perplexity <= th ? 1 : 0;
}

int wubu_evictc_block_paged(const int *kv_table, int n_blocks, int block_size, int *to_evict)
{
    if (!kv_table || !to_evict || n_blocks <= 0) return -1;
    /* evict the first block (simplification of the policy) */
    *to_evict = 0;
    return 0;
}

int wubu_evictc_batch(const float *criticality, int n, float th, int *evicted)
{
    if (!criticality || !evicted || n <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n; i++)
        if (criticality[i] < th) evicted[k++] = i;
    return k;
}

int wubu_evictc_kvquant_kernel(const float *kv, int n, int8_t *out)
{
    if (!kv || !out) return -1;
    for (int i = 0; i < n; i++) {
        float v = kv[i] < -1 ? -1 : (kv[i] > 1 ? 1 : kv[i]);
        out[i] = (int8_t)roundf(v * 3.0f);
    }
    return n;
}

int wubu_evictc_ann(const float *sim_scores, int n, float th, int *keep)
{
    if (!sim_scores || !keep || n <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n; i++)
        if (sim_scores[i] >= th) keep[k++] = i;
    return k;
}

int wubu_evictc_spec(const float *draft_scores, int n, float th, int *retain)
{
    if (!draft_scores || !retain || n <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n; i++)
        if (draft_scores[i] >= th) retain[k++] = i;
    return k;
}

int wubu_evictc_scaling(float *attn, int n, float factor)
{
    if (!attn || n <= 0) return -1;
    for (int i = 0; i < n; i++) attn[i] *= factor;
    return 0;
}

int wubu_evictc_1m(long ctx_size, long threshold)
{
    return ctx_size > threshold ? 1 : 0;
}

int wubu_evictc_hybrid(const float *attn_scores, const float *ssm_scores, int n, float th)
{
    if (!attn_scores || !ssm_scores || n <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n; i++)
        if (attn_scores[i] >= th && ssm_scores[i] >= th) k++;
    return k;
}

int wubu_evictc_mm(const float *vision_scores, const float *text_scores, int n, float th)
{
    if (!vision_scores || !text_scores || n <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n; i++)
        if (vision_scores[i] >= th || text_scores[i] >= th) k++;
    return k;
}