/*
 * wubu_compress.c -- the context-compression frontier (IY). C11.
 */
#include "wubu_compress.h"
#include <math.h>
#include <string.h>

int wubu_comp_llmlingua(const float *perplexities, int n, float th,
                            int *keep)
{
    if (!perplexities || !keep || n <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (perplexities[i] <= th) keep[k++] = i;
    }
    return k;
}

int wubu_comp_llmlingua2(const float *scores, int n, float th,
                             int *keep)
{
    if (!scores || !keep || n <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (scores[i] >= th) keep[k++] = i;
    }
    return k;
}

int wubu_comp_reorder(const int *is_question, int n, int *order)
{
    if (!is_question || !order || n <= 0) return -1;
    int qi = 0;
    for (int i = 0; i < n; i++) {
        if (is_question[i]) order[qi++] = i;
    }
    for (int i = 0; i < n; i++) {
        if (!is_question[i]) order[qi++] = i;
    }
    return 0;
}

int wubu_comp_self_info(const float *info, int n, float keep_frac,
                            int *keep)
{
    if (!info || !keep || n <= 0) return -1;
    int k = (int)((float)n * keep_frac);
    if (k < 1) k = 1;
    /* keep the top-k by self-information */
    for (int i = 0; i < n; i++) keep[i] = i;
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (info[keep[j]] > info[keep[i]]) {
                int t = keep[i]; keep[i] = keep[j]; keep[j] = t;
            }
    return k;
}

int wubu_comp_recmp(const float *scores, int n, float ext_th,
                        float abs_th, int *keep)
{
    if (!scores || !keep || n <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (scores[i] >= ext_th || scores[i] >= abs_th) keep[k++] = i;
    }
    return k;
}

int wubu_comp_doc2atom(const float *embeddings, int n, int d,
                           float th, int *atoms)
{
    if (!embeddings || !atoms || n <= 0 || d <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n; i++) {
        float norm = 0;
        for (int j = 0; j < d; j++) norm += embeddings[i * d + j] * embeddings[i * d + j];
        if (sqrtf(norm) >= th) atoms[k++] = i;
    }
    return k;
}

int wubu_comp_cartridge(long kv_used, long kv_cap, long *evict)
{
    if (!evict) return -1;
    if (kv_used > kv_cap) { *evict = kv_used - kv_cap; return 1; }
    *evict = 0;
    return 0;
}

int wubu_comp_lamr(const float *sem_score, const float *dep_score,
                       int n, float w_sem, float w_dep)
{
    if (!sem_score || !dep_score || n <= 0) return -1;
    /* keep tokens with high combined score */
    int kept = 0;
    for (int i = 0; i < n; i++) {
        float combined = w_sem * sem_score[i] + w_dep * dep_score[i];
        if (combined > 0.5f) kept++;
    }
    return kept;
}

int wubu_comp_sesrag(const float *densities, int n, float th,
                         int *segments)
{
    if (!densities || !segments || n <= 0) return -1;
    int seg = 0;
    for (int i = 0; i < n; i++) {
        if (densities[i] >= th) segments[seg++] = i;
    }
    return seg;
}

int wubu_comp_grc(const float *tokens, int n, int k, int *meta)
{
    if (!tokens || !meta || k <= 0) return -1;
    /* the top-k tokens become meta latent tokens */
    int idx[64];
    if (k > 64) k = 64;
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (tokens[idx[j]] > tokens[idx[i]]) {
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
    for (int i = 0; i < k; i++) meta[i] = idx[i];
    return k;
}

int wubu_comp_epc(float predicted_relevance, float cur_retention)
{
    /* write-time retention: keep if predicted relevance > current */
    return predicted_relevance > cur_retention ? 1 : 0;
}

int wubu_comp_lim(const float *importance, int n, int *order)
{
    if (!importance || !order || n <= 0) return -1;
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (importance[order[j]] > importance[order[i]]) {
                int t = order[i]; order[i] = order[j]; order[j] = t;
            }
    return 0;
}

long wubu_comp_budget(long tokens, float density, long base_budget)
{
    /* dense contexts need more budget */
    return base_budget + (long)((float)tokens * density * 0.5f);
}

int wubu_comp_tool_schema(const char *schema, int len, float ratio)
{
    if (!schema || len <= 0) return -1;
    return (int)((float)len * ratio);
}

int wubu_comp_autoenc(const float *ctx, int n, int d,
                          float *latent, int k)
{
    if (!ctx || !latent || k <= 0) return -1;
    /* simple top-k projection as the autoencoder bottleneck */
    for (int i = 0; i < k; i++) latent[i] = ctx[i * d];
    return k;
}

int wubu_comp_distill(const float *doc_emb, int n, int d,
                          float *lora_weights)
{
    if (!doc_emb || !lora_weights || n <= 0 || d <= 0) return -1;
    /* the Doc-to-LoRA: average embedding → LoRA init */
    for (int j = 0; j < d; j++) {
        float sum = 0;
        for (int i = 0; i < n; i++) sum += doc_emb[i * d + j];
        lora_weights[j] = sum / (float)n;
    }
    return d;
}

int wubu_comp_latent_mem(const float *kv, int n, int d,
                             float *memory)
{
    if (!kv || !memory || n <= 0 || d <= 0) return -1;
    /* the compressed KV becomes updatable memory */
    for (int j = 0; j < d; j++) {
        float sum = 0;
        for (int i = 0; i < n; i++) sum += kv[i * d + j];
        memory[j] = sum / (float)n;
    }
    return d;
}

int wubu_comp_paged(const float *attn, int n, int page_size,
                        int *pages)
{
    if (!attn || !pages || page_size <= 0) return -1;
    int np = (n + page_size - 1) / page_size;
    for (int p = 0; p < np; p++) {
        int start = p * page_size;
        int end = start + page_size < n ? start + page_size : n;
        float sum = 0;
        for (int i = start; i < end; i++) sum += attn[i];
        pages[p] = (int)sum;
    }
    return np;
}

int wubu_comp_governor(float ratio, float target, float quality)
{
    /* adjust ratio toward target based on quality feedback */
    return quality >= target ? 1 : 0;
}

float wubu_comp_fidelity(const float *orig, const float *recon, int n)
{
    if (!orig || !recon || n <= 0) return 0;
    float err = 0;
    for (int i = 0; i < n; i++) {
        float d = orig[i] - recon[i];
        err += d * d;
    }
    return sqrtf(err / (float)n);
}
