/*
 * wubu_vision.c -- the multimodal vision frontier (JB). C11.
 */
#include "wubu_vision.h"
#include <math.h>
#include <string.h>

int wubu_vision_selector(const float *scores, int n, float th, int *keep)
{
    if (!scores || !keep || n <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n; i++)
        if (scores[i] >= th) keep[k++] = i;
    return k;
}

float wubu_vision_text_eff(int text_tokens, int pixel_tokens)
{
    if (text_tokens <= 0) return 0;
    return (float)pixel_tokens / (float)text_tokens;
}

int wubu_vision_img_compress(int patches, int merge_factor, int *out)
{
    if (!out || merge_factor <= 1) return -1;
    *out = (patches + merge_factor - 1) / merge_factor;
    return 0;
}

int wubu_vision_vid_compress(int frames, int fps, float redundancy, int *out)
{
    if (!out || fps <= 0 || redundancy < 0 || redundancy > 1) return -1;
    *out = (int)((float)frames * (1.0f - redundancy));
    return 0;
}

int wubu_vision_audio_compress(int spec_bins, float redundancy, int *out)
{
    if (!out || spec_bins <= 0) return -1;
    *out = (int)((float)spec_bins * (1.0f - redundancy));
    return 0;
}

int wubu_vision_clip_align(const float *vis, const float *txt, int d,
                                float *sim)
{
    if (!vis || !txt || !sim) return -1;
    float dot = 0, vn = 0, tn = 0;
    for (int i = 0; i < d; i++) {
        dot += vis[i] * txt[i];
        vn += vis[i] * vis[i];
        tn += txt[i] * txt[i];
    }
    *sim = dot / (sqrtf(vn) * sqrtf(tn) + 1e-9f);
    return 0;
}

int wubu_vision_redundancy(const float *patches, int n, int d,
                                float th, int *keep)
{
    if (!patches || !keep || n <= 0) return -1;
    int k = 0;
    keep[k++] = 0;
    for (int i = 1; i < n; i++) {
        int dup = 0;
        for (int j = 0; j < k; j++) {
            float dist = 0;
            for (int x = 0; x < d; x++) {
                float diff = patches[i * d + x] - patches[keep[j] * d + x];
                dist += diff * diff;
            }
            if (sqrtf(dist) < th) { dup = 1; break; }
        }
        if (!dup) keep[k++] = i;
    }
    return k;
}

int wubu_vision_kv_budget(int vis_kv, int txt_kv, int total, int *alloc)
{
    if (!alloc) return -1;
    int sum = vis_kv + txt_kv;
    if (sum <= total) { alloc[0] = vis_kv; alloc[1] = txt_kv; return 0; }
    float ratio = (float)vis_kv / (float)sum;
    alloc[0] = (int)((float)total * ratio);
    alloc[1] = total - alloc[0];
    return 0;
}

int wubu_vision_sparse(const float *attn, int n, float th, int *keep)
{
    if (!attn || !keep || n <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n; i++)
        if (attn[i] >= th) keep[k++] = i;
    return k;
}

int wubu_vision_importance(const float *features, int n, int k, int *topk)
{
    if (!features || !topk || n <= 0 || k <= 0) return -1;
    if (k > n) k = n;
    for (int i = 0; i < n; i++) topk[i] = i;
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (features[topk[j]] > features[topk[i]]) {
                int t = topk[i]; topk[i] = topk[j]; topk[j] = t;
            }
    return k;
}

int wubu_vision_av_fusion(const float *audio, const float *vis, int n,
                               float *fused)
{
    if (!audio || !vis || !fused) return -1;
    for (int i = 0; i < n; i++)
        fused[i] = 0.5f * audio[i] + 0.5f * vis[i];
    return n;
}

int wubu_vision_budget_plan(int vis_tok, int txt_tok, long total_budget,
                                 int *vis_alloc, int *txt_alloc)
{
    if (!vis_alloc || !txt_alloc) return -1;
    int total = vis_tok + txt_tok;
    if (total <= total_budget) { *vis_alloc = vis_tok; *txt_alloc = txt_tok; return 0; }
    float ratio = (float)vis_tok / (float)total;
    *vis_alloc = (int)((float)total_budget * ratio);
    *txt_alloc = total_budget - *vis_alloc;
    return 0;
}

float wubu_vision_enc_eff(int patches, int d_model)
{
    if (patches <= 0 || d_model <= 0) return 0;
    return (float)d_model / (float)patches;
}

int wubu_vision_evict(const float *salience, int n, float th, int *evict)
{
    if (!salience || !evict || n <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n; i++)
        if (salience[i] < th) evict[k++] = i;
    return k;
}

int wubu_vision_prefix(const float *vis_prefix, const float *txt_prefix,
                            int d, float *shared)
{
    if (!vis_prefix || !txt_prefix || !shared) return -1;
    for (int i = 0; i < d; i++)
        shared[i] = 0.5f * (vis_prefix[i] + txt_prefix[i]);
    return d;
}

int wubu_vision_stream(const float *tokens, int n, int d, int window,
                            float *out)
{
    if (!tokens || !out || window <= 0) return -1;
    int out_n = n < window ? n : window;
    for (int i = 0; i < out_n; i++)
        for (int j = 0; j < d; j++)
            out[i * d + j] = tokens[i * d + j];
    return out_n;
}

float wubu_vision_energy(int modality, long tokens, float j_per_token)
{
    return (float)tokens * j_per_token;
}

int wubu_vision_dedup(const float *tokens, int n, int d, float th, int *keep)
{
    if (!tokens || !keep || n <= 0) return -1;
    int k = 0;
    keep[k++] = 0;
    for (int i = 1; i < n; i++) {
        int dup = 0;
        for (int j = 0; j < k; j++) {
            float dist = 0;
            for (int x = 0; x < d; x++) {
                float diff = tokens[i * d + x] - tokens[keep[j] * d + x];
                dist += diff * diff;
            }
            if (sqrtf(dist) < th) { dup = 1; break; }
        }
        if (!dup) keep[k++] = i;
    }
    return k;
}

int wubu_vision_route(const float *task_vec, int n, float *weights)
{
    if (!task_vec || !weights || n <= 0) return -1;
    float sum = 0;
    for (int i = 0; i < n; i++) sum += task_vec[i];
    for (int i = 0; i < n; i++) weights[i] = sum > 0 ? task_vec[i] / sum : 1.0f / n;
    return 0;
}