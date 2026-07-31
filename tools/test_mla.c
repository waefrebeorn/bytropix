/* Test: MLA Multi-head Latent Attention (doc E02). */
#include "wubu_mla.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

int main(void) {
    int hidden_dim = 256, n_heads = 4, head_dim = 64;
    int q_lora_rank = 32, kv_lora_rank = 16, rope_head_dim = 8;

    wubu_mla_t *m = wubu_mla_create(hidden_dim, n_heads, head_dim,
                                     q_lora_rank, kv_lora_rank, rope_head_dim);
    assert(m);
    assert(m->kv_latent_dim == kv_lora_rank + rope_head_dim);
    printf("MLA created: hidden=%d heads=%d dim=%d kv_latent=%d\n",
           m->hidden_dim, m->n_heads, m->head_dim, m->kv_latent_dim);

    /* Test 1: compression ratio — MLA KV is ~14x smaller than standard */
    float ratio = wubu_mla_compression_ratio(m);
    float standard = 2.0f * n_heads * head_dim;
    float mla_kv = (float)m->kv_latent_dim;
    printf("Standard KV/token: %.0f floats, MLA KV/token: %.0f floats, ratio: %.2fx\n",
           (double)standard, (double)mla_kv, (double)ratio);
    assert(ratio > 1.0f);

    /* Test 2: down-project KV */
    float *x = (float *)malloc(hidden_dim * sizeof(float));
    float *kv_latent = (float *)malloc(m->kv_latent_dim * sizeof(float));
    float *W_DKV = (float *)malloc((size_t)m->kv_latent_dim * hidden_dim * sizeof(float));
    for (int i = 0; i < hidden_dim; i++) x[i] = 0.01f * i;
    for (int i = 0; i < (int)(m->kv_latent_dim * hidden_dim); i++) W_DKV[i] = 0.001f * (i % 17 - 8);

    wubu_mla_down_proj_kv(m, W_DKV, x, kv_latent);
    for (int i = 0; i < m->kv_lora_rank; i++)
        assert(!isnan(kv_latent[i]) && !isinf(kv_latent[i]));
    printf("Down-project KV: %d → %d (latent) ✓\n", hidden_dim, m->kv_latent_dim);

    /* Test 3: up-project K and V */
    int total_heads = n_heads * head_dim;
    float *k_nope = (float *)malloc(total_heads * sizeof(float));
    float *v_proj = (float *)malloc(total_heads * sizeof(float));
    float *W_UK = (float *)malloc((size_t)total_heads * m->kv_lora_rank * sizeof(float));
    float *W_UV = (float *)malloc((size_t)total_heads * m->kv_lora_rank * sizeof(float));
    for (int i = 0; i < (int)(total_heads * m->kv_lora_rank); i++) {
        W_UK[i] = 0.001f * (i % 13 - 6);
        W_UV[i] = 0.001f * (i % 11 - 5);
    }

    wubu_mla_up_proj_k(m, W_UK, kv_latent, k_nope);
    wubu_mla_up_proj_v(m, W_UV, kv_latent, v_proj);
    for (int i = 0; i < total_heads; i++) {
        assert(!isnan(k_nope[i]));
        assert(!isnan(v_proj[i]));
    }
    printf("Up-project K+V: %d → %d (per head) ✓\n", m->kv_lora_rank, total_heads);

    /* Test 4: project Q via LoRA */
    float *q = (float *)malloc(total_heads * sizeof(float));
    float *W_DQ = (float *)malloc((size_t)m->q_lora_rank * hidden_dim * sizeof(float));
    float *W_UQ = (float *)malloc((size_t)total_heads * m->q_lora_rank * sizeof(float));
    for (int i = 0; i < (int)(m->q_lora_rank * hidden_dim); i++) W_DQ[i] = 0.001f * (i % 7 - 3);
    for (int i = 0; i < (int)(total_heads * m->q_lora_rank); i++) W_UQ[i] = 0.001f * (i % 9 - 4);

    wubu_mla_proj_q(m, W_DQ, W_UQ, x, q);
    for (int i = 0; i < total_heads; i++) assert(!isnan(q[i]));
    printf("Project Q via LoRA: %d → %d → %d ✓\n", hidden_dim, m->q_lora_rank, total_heads);

    /* Test 5: attention computation */
    float *k_rope = (float *)malloc(n_heads * rope_head_dim * sizeof(float));
    float *out = (float *)malloc(total_heads * sizeof(float));
    for (int i = 0; i < n_heads * rope_head_dim; i++) k_rope[i] = 0.1f;
    memset(out, 0, total_heads * sizeof(float));

    wubu_mla_attn(m, q, k_nope, k_rope, v_proj, out);
    for (int i = 0; i < total_heads; i++) assert(!isnan(out[i]));
    printf("MLA attention: computed %d head outputs ✓\n", n_heads);

    free(x); free(kv_latent); free(W_DKV); free(k_nope); free(v_proj);
    free(W_UK); free(W_UV); free(q); free(W_DQ); free(W_UQ);
    free(k_rope); free(out);
    wubu_mla_free(m);
    printf("ALL MLA TESTS PASSED\n");
    return 0;
}
