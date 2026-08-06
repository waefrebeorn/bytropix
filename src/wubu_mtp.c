/* wubu_mtp.c — Multi-Token Prediction head (speculative decode).
 * Extracted from wubu_model.c per ADR-002 opaque-struct seam.
 * MTP is an optional head at layer 40; the main model loads it
 * after wubu_model_init when the GGUF contains nextn.* tensors.
 */

#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Load the MTP head from the same GGUF file as the main model.
 * Must be called after wubu_model_init.  The main model's gguf_ctx
 * and data_blob are reused — no second file open. */
bool wubu_mtp_load(mtp_head_t *mtp, const char *mtp_gguf_path,
                   gguf_ctx *main_ctx, const uint8_t *main_blob,
                   int gqa_max_ctx)
{
    memset(mtp, 0, sizeof(*mtp));

    gguf_ctx *ctx = main_ctx;
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    if (!ctx || !blob) {
        fprintf(stderr, "MTP: no context or blob available\n");
        return false;
    }

    /* Verify this is an MTP model (has nextn tensors at layer 40). */
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.40.nextn.hnorm.weight");
    if (!t) {
        fprintf(stderr, "MTP: no nextn tensors in model (not an MTP model?)\n");
        return false;
    }

    /* Nextn norms (F32, D_MODEL). */
    t = gguf_find_tensor(ctx, "blk.40.nextn.hnorm.weight");
    if (!t) { fprintf(stderr, "MTP: missing hnorm\n"); goto fail; }
    mtp->nextn_hnorm = (float *)malloc(D_MODEL * sizeof(float));
    gguf_read_tensor_f32(ctx, t, mtp->nextn_hnorm, D_MODEL);

    t = gguf_find_tensor(ctx, "blk.40.nextn.enorm.weight");
    if (!t) { fprintf(stderr, "MTP: missing enorm\n"); goto fail; }
    mtp->nextn_enorm = (float *)malloc(D_MODEL * sizeof(float));
    gguf_read_tensor_f32(ctx, t, mtp->nextn_enorm, D_MODEL);

    t = gguf_find_tensor(ctx, "blk.40.nextn.shared_head_norm.weight");
    if (!t) { fprintf(stderr, "MTP: missing shared_head_norm\n"); goto fail; }
    mtp->nextn_shared_head_norm = (float *)malloc(D_MODEL * sizeof(float));
    gguf_read_tensor_f32(ctx, t, mtp->nextn_shared_head_norm, D_MODEL);

    /* eh_proj weight — dequant Q8_0 to F32 during init for fast SGEMM. */
    t = gguf_find_tensor(ctx, "blk.40.nextn.eh_proj.weight");
    if (!t) { fprintf(stderr, "MTP: missing eh_proj\n"); goto fail; }
    mtp->nextn_eh_proj_dim = (int64_t)t->dims[0];
    int64_t eh_elems = (int64_t)t->dims[0] * (int64_t)t->dims[1];
    mtp->nextn_eh_proj_f32 = (float *)malloc(eh_elems * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, mtp->nextn_eh_proj_f32, eh_elems)) {
        fprintf(stderr, "MTP: failed to read eh_proj\n"); goto fail;
    }
    printf("MTP: eh_proj dequantized (%lld x %lld = %lld elems)\n",
           (long long)t->dims[0], (long long)t->dims[1], (long long)eh_elems);

    /* Blk.40 GQA norms (F32). */
    t = gguf_find_tensor(ctx, "blk.40.attn_norm.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_norm\n"); goto fail; }
    mtp->blk40.attn_norm_weight = (float *)malloc(D_MODEL * sizeof(float));
    gguf_read_tensor_f32(ctx, t, mtp->blk40.attn_norm_weight, D_MODEL);

    t = gguf_find_tensor(ctx, "blk.40.post_attention_norm.weight");
    if (!t) { fprintf(stderr, "MTP: missing post_attn_norm\n"); goto fail; }
    mtp->blk40.post_attn_norm_weight = (float *)malloc(D_MODEL * sizeof(float));
    gguf_read_tensor_f32(ctx, t, mtp->blk40.post_attn_norm_weight, D_MODEL);

    /* Blk.40 GQA weights (Q5_K — type 13). */
    t = gguf_find_tensor(ctx, "blk.40.attn_q.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_q\n"); goto fail; }
    mtp->blk40.gqa.attn_q_weight_q = blob + t->data_offset;
    mtp->blk40.gqa.attn_q_weight_type = t->ggml_type;

    t = gguf_find_tensor(ctx, "blk.40.attn_k.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_k\n"); goto fail; }
    mtp->blk40.gqa.attn_k_weight_q = blob + t->data_offset;
    mtp->blk40.gqa.attn_k_weight_type = t->ggml_type;

    t = gguf_find_tensor(ctx, "blk.40.attn_v.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_v\n"); goto fail; }
    mtp->blk40.gqa.attn_v_weight_q = blob + t->data_offset;
    mtp->blk40.gqa.attn_v_weight_type = t->ggml_type;

    t = gguf_find_tensor(ctx, "blk.40.attn_output.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_output\n"); goto fail; }
    mtp->blk40.gqa.attn_output_weight_q = blob + t->data_offset;
    mtp->blk40.gqa.attn_output_weight_type = t->ggml_type;

    /* Q/K norms (F32). */
    t = gguf_find_tensor(ctx, "blk.40.attn_q_norm.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_q_norm\n"); goto fail; }
    int mtp_head_dim = (t->n_dims >= 1) ? (int)t->dims[0] : GQA_HEAD_DIM;
    mtp->blk40.gqa.head_dim = mtp_head_dim;
    mtp->blk40.gqa.attn_q_norm_weight = (float *)malloc(mtp_head_dim * sizeof(float));
    gguf_read_tensor_f32(ctx, t, mtp->blk40.gqa.attn_q_norm_weight, mtp_head_dim);

    t = gguf_find_tensor(ctx, "blk.40.attn_k_norm.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_k_norm\n"); goto fail; }
    mtp->blk40.gqa.attn_k_norm_weight = (float *)malloc(mtp_head_dim * sizeof(float));
    gguf_read_tensor_f32(ctx, t, mtp->blk40.gqa.attn_k_norm_weight, mtp_head_dim);

    mtp->blk40.gqa.q_heads = GQA_Q_HEADS;
    mtp->blk40.gqa.kv_heads = GQA_KV_HEADS;
    mtp->blk40.gqa.q_dim = GQA_Q_HEADS * mtp_head_dim;
    mtp->blk40.gqa.kv_dim = GQA_KV_HEADS * mtp_head_dim;

    /* Blk.40 MoE weights (quantized blob pointers). */
    moe_weights_t *moe = &mtp->blk40.moe;

    t = gguf_find_tensor(ctx, "blk.40.ffn_gate_inp.weight");
    if (t && blob) {
        int64_t n_router = (int64_t)t->dims[0] * t->dims[1];
        moe->ffn_gate_inp = (float *)malloc(n_router * sizeof(float));
        gguf_read_tensor_f32(ctx, t, moe->ffn_gate_inp, n_router);
    }

    t = gguf_find_tensor(ctx, "blk.40.ffn_gate_inp_shexp.weight");
    if (t && blob) {
        moe->ffn_gate_inp_shexp = (float *)malloc(D_MODEL * sizeof(float));
        gguf_read_tensor_f32(ctx, t, moe->ffn_gate_inp_shexp, D_MODEL);
    }

    /* Routed experts: Q2_K (gate, up), Q3_K (down). */
    t = gguf_find_tensor(ctx, "blk.40.ffn_gate_exps.weight");
    if (t && blob) { moe->ffn_gate_exps_q = blob + t->data_offset; moe->ffn_gate_exps_q_type = t->ggml_type; }
    t = gguf_find_tensor(ctx, "blk.40.ffn_up_exps.weight");
    if (t && blob) { moe->ffn_up_exps_q = blob + t->data_offset; moe->ffn_up_exps_q_type = t->ggml_type; }
    t = gguf_find_tensor(ctx, "blk.40.ffn_down_exps.weight");
    if (t && blob) { moe->ffn_down_exps_q = blob + t->data_offset; moe->ffn_down_exps_q_type = t->ggml_type; }

    /* Shared expert: Q5_K (gate, up), Q6_K (down). */
    t = gguf_find_tensor(ctx, "blk.40.ffn_gate_shexp.weight");
    if (t && blob) { moe->ffn_gate_shexp_q = blob + t->data_offset; moe->ffn_gate_shexp_q_type = t->ggml_type; }
    t = gguf_find_tensor(ctx, "blk.40.ffn_up_shexp.weight");
    if (t && blob) { moe->ffn_up_shexp_q = blob + t->data_offset; moe->ffn_up_shexp_q_type = t->ggml_type; }
    t = gguf_find_tensor(ctx, "blk.40.ffn_down_shexp.weight");
    if (t && blob) { moe->ffn_down_shexp_q = blob + t->data_offset; moe->ffn_down_shexp_q_type = t->ggml_type; }

    if (moe->ffn_gate_exps_q && moe->ffn_up_exps_q && moe->ffn_down_exps_q) {
        moe->loaded = true;
        moe->load_from_blob = true;
    }

    printf("MTP: blk.40 loaded (GQA+MoE: Q5_K/Q2_K/Q3_K/Q6_K)\n");

    /* KV cache for blk.40. */
    int mtp_kv_dim = mtp->blk40.gqa.kv_heads * mtp->blk40.gqa.head_dim;
    mtp->kv_dim = mtp_kv_dim;
    mtp->k_cache = (float *)calloc((size_t)gqa_max_ctx * mtp_kv_dim, sizeof(float));
    mtp->v_cache = (float *)calloc((size_t)gqa_max_ctx * mtp_kv_dim, sizeof(float));
    mtp->cache_len = 0;

    mtp->loaded = true;
    return true;

fail:
    wubu_mtp_free(mtp);
    return false;
}

/* Draft forward: predict next tokens from the last hidden state.
 * x: [D_MODEL] — last hidden state from main model (layer 39 output).
 * token_embd: [B, D_MODEL] — embeddings of candidate continuation tokens.
 * B: number of draft candidates.
 * logits_out: [B, vocab_size] — output logits per candidate.
 * Returns: number of tokens consumed (for KV cache tracking). */
int wubu_mtp_draft_forward(wubu_model_t *model,
                           const float *x,
                           const float *token_embd, int B,
                           float *logits_out)
{
    if (!model->mtp.loaded) return 0;

    mtp_head_t *mtp = &model->mtp;
    wubu_layer_t *blk40 = &mtp->blk40;
    const int vs = model->vocab_size;

    float *h_norm = (float *)malloc(model->d_model * sizeof(float));
    float *e_norm = (float *)malloc(model->d_model * sizeof(float));
    float *concat = (float *)malloc(2 * model->d_model * sizeof(float));
    float *cur = (float *)malloc(model->d_model * sizeof(float));
    float *temp_attn = (float *)malloc(model->d_model * sizeof(float));
    float *temp_ffn = (float *)malloc(model->d_model * sizeof(float));
    float *temp_norm = (float *)malloc(model->d_model * sizeof(float));

    if (!h_norm || !e_norm || !concat || !cur || !temp_attn || !temp_ffn || !temp_norm) {
        fprintf(stderr, "MTP draft: alloc failed\n");
        free(h_norm); free(e_norm); free(concat); free(cur);
        free(temp_attn); free(temp_ffn); free(temp_norm);
        return 0;
    }

    /* Step 1: h_norm = rms_norm(x, hnorm) */
    wubu_rms_norm(1, 1, model->d_model, x, mtp->nextn_hnorm, 1e-6f, h_norm);

    for (int b = 0; b < B; b++) {
        const float *embd_b = token_embd + b * model->d_model;
        float *logits_b = logits_out + b * vs;

        /* Step 2: e_norm = rms_norm(token_embd[b], enorm) */
        wubu_rms_norm(1, 1, model->d_model, embd_b, mtp->nextn_enorm, 1e-6f, e_norm);

        /* Step 3: concat = [e_norm | h_norm] (llama.cpp order). */
        memcpy(concat, e_norm, model->d_model * sizeof(float));
        memcpy(concat + model->d_model, h_norm, model->d_model * sizeof(float));

        /* Step 4: cur = eh_proj @ concat (F32 SGEMM). */
        for (int j = 0; j < model->d_model; j++) {
            double sum = 0.0;
            for (int k = 0; k < mtp->nextn_eh_proj_dim; k++)
                sum += (double)concat[k] * (double)mtp->nextn_eh_proj_f32[j * mtp->nextn_eh_proj_dim + k];
            cur[j] = (float)sum;
        }

        /* Step 5: Forward through blk.40 (GQA+MoE). */
        wubu_rms_norm(1, 1, model->d_model, cur, blk40->attn_norm_weight, 1e-6f, temp_norm);

        float *k_out = mtp->k_cache + (size_t)(mtp->cache_len + b) * mtp->kv_dim;
        float *v_out = mtp->v_cache + (size_t)(mtp->cache_len + b) * mtp->kv_dim;
        wubu_gqa_forward(temp_norm, 1, 1, &blk40->gqa, model->d_model, temp_attn,
                         mtp->k_cache, mtp->v_cache, mtp->cache_len + b,
                         k_out, v_out,
                         blk40->gqa.head_dim, blk40->gqa.q_heads, blk40->gqa.kv_heads);

        for (int i = 0; i < model->d_model; i++) cur[i] += temp_attn[i];

        wubu_rms_norm(1, 1, model->d_model, cur, blk40->post_attn_norm_weight, 1e-6f, temp_norm);

        if (blk40->moe.loaded) {
            wubu_moe_forward(temp_norm, 1, 1, &blk40->moe, temp_ffn, NULL,
                             model->n_active_experts, model->n_experts, model->d_model, model->d_ff);
        } else {
            memcpy(temp_ffn, temp_norm, model->d_model * sizeof(float));
        }

        for (int i = 0; i < model->d_model; i++) cur[i] += temp_ffn[i];

        /* Step 6: shared_head_norm */
        wubu_rms_norm(1, 1, model->d_model, cur, mtp->nextn_shared_head_norm, 1e-6f, temp_norm);

        /* Step 7: output projection (via main model's output.weight). */
        if (model->output_weight_q) {
            quantized_matmul(temp_norm, model->output_weight_q, model->output_weight_type,
                             model->d_model, vs, 0, logits_b);
        } else {
            memset(logits_b, 0, vs * sizeof(float));
        }
    }

    mtp->cache_len += B;

    free(h_norm); free(e_norm); free(concat); free(cur);
    free(temp_attn); free(temp_ffn); free(temp_norm);

    return B;
}

/* Free MTP head resources. */
void wubu_mtp_free(mtp_head_t *mtp)
{
    if (!mtp || !mtp->loaded) return;
    free(mtp->nextn_hnorm);
    free(mtp->nextn_enorm);
    free(mtp->nextn_shared_head_norm);
    free(mtp->nextn_eh_proj_f32);
    free(mtp->blk40.attn_norm_weight);
    free(mtp->blk40.post_attn_norm_weight);
    free(mtp->blk40.gqa.attn_q_norm_weight);
    free(mtp->blk40.gqa.attn_k_norm_weight);
    /* MoE blob-backed: only F32 pointers freed. */
    free(mtp->blk40.moe.ffn_gate_inp);
    free(mtp->blk40.moe.ffn_gate_inp_shexp);
    free(mtp->k_cache);
    free(mtp->v_cache);
    memset(mtp, 0, sizeof(*mtp));
}
