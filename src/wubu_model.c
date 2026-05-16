#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ========== GGUF Tensor Names ==========

static const char *tensor_name_attn_norm(int layer) {
    static char buf[64];
    snprintf(buf, sizeof(buf), "blk.%d.attn_norm.weight", layer);
    return buf;
}

static const char *tensor_name_post_attn_norm(int layer) {
    static char buf[64];
    snprintf(buf, sizeof(buf), "blk.%d.post_attention_norm.weight", layer);
    return buf;
}

// ========== Init ==========

bool wubu_model_init(wubu_model_t *model, const char *gguf_path) {
    memset(model, 0, sizeof(*model));
    
    // Open GGUF
    gguf_ctx *ctx = gguf_open(gguf_path);
    if (!ctx) { fprintf(stderr, "Failed to open %s\n", gguf_path); return false; }
    
    // Count layers from tensor names
    // Find max layer index from any blk.N. tensor
    int max_layer = 0;
    for (int i = 0; i < (int)ctx->n_tensors; i++) {
        const char *name = ctx->tensors[i].name;
        if (strncmp(name, "blk.", 4) == 0) {
            int layer = atoi(name + 4);
            if (layer > max_layer) max_layer = layer;
        }
    }
    model->n_layers = max_layer + 1;  // 40 layers for Qwen3.6
    
    // Allocate layers
    model->layers = (wubu_layer_t *)calloc(model->n_layers, sizeof(wubu_layer_t));
    if (!model->layers) { gguf_close(ctx); return false; }
    
    printf("Allocating %d layers...\n", model->n_layers);
    
    // Load layer norms and attention weights
    for (int l = 0; l < model->n_layers; l++) {
        wubu_layer_t *layer = &model->layers[l];
        layer->layer_idx = l;
        layer->is_ssm = wubu_is_ssm_layer(l);
        
        gguf_tensor_info *t;
        
        // attn_norm.weight (pre-attention RMSNorm)
        t = gguf_find_tensor(ctx, tensor_name_attn_norm(l));
        if (t) {
            layer->attn_norm_weight = (float *)malloc(D_MODEL * sizeof(float));
            if (!gguf_read_tensor_f32(ctx, t, layer->attn_norm_weight, D_MODEL))
                { fprintf(stderr, "Failed to load attn_norm[%d]\n", l); goto fail; }
        }
        
        // post_attention_norm.weight
        t = gguf_find_tensor(ctx, tensor_name_post_attn_norm(l));
        if (t) {
            layer->post_attn_norm_weight = (float *)malloc(D_MODEL * sizeof(float));
            if (!gguf_read_tensor_f32(ctx, t, layer->post_attn_norm_weight, D_MODEL))
                { fprintf(stderr, "Failed to load post_attn_norm[%d]\n", l); goto fail; }
        }
        
        if (layer->is_ssm) {
            // Load SSM weights — reuses load_ssm_layer logic inline
            char name[256];
            int qkv_dim = KEY_DIM * 2 + VALUE_DIM;
            int ok = 1;
            
            snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->ssm.attn_qkv_weight = (float *)malloc(D_MODEL * qkv_dim * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.attn_qkv_weight, D_MODEL * qkv_dim) > 0);
            
            snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->ssm.attn_gate_weight = (float *)malloc(D_MODEL * VALUE_DIM * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.attn_gate_weight, D_MODEL * VALUE_DIM) > 0);
            
            snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->ssm.ssm_beta_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_beta_weight, D_MODEL * DT_RANK) > 0);
            
            snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->ssm.ssm_alpha_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_alpha_weight, D_MODEL * DT_RANK) > 0);
            
            snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->ssm.ssm_dt_bias = (float *)malloc(DT_RANK * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_dt_bias, DT_RANK) > 0);
            
            snprintf(name, sizeof(name), "blk.%d.ssm_a", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->ssm.ssm_a = (float *)malloc(DT_RANK * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_a, DT_RANK) > 0);
            
            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->ssm.ssm_conv1d_weight = (float *)malloc(CONV_KERNEL * CONV_DIM * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_conv1d_weight, CONV_KERNEL * CONV_DIM) > 0);
            
            snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->ssm.ssm_norm_weight = (float *)malloc(SSM_D_STATE * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_norm_weight, SSM_D_STATE) > 0);
            
            snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->ssm.ssm_out_weight = (float *)malloc(VALUE_DIM * D_MODEL * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_out_weight, VALUE_DIM * D_MODEL) > 0);
            
            if (!ok) { fprintf(stderr, "Failed to load SSM weights for layer %d\n", l); goto fail; }
            printf("  Layer %d: SSM loaded\n", l);
            
        } else {
            // Load GQA weights
            char name[256];
            int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
            int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
            int ok = 1;
            
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->gqa.attn_q_weight = (float *)malloc(D_MODEL * q_dim * 2 * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->gqa.attn_q_weight, D_MODEL * q_dim * 2) > 0);
            
            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->gqa.attn_k_weight = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->gqa.attn_k_weight, D_MODEL * kv_dim) > 0);
            
            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->gqa.attn_v_weight = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->gqa.attn_v_weight, D_MODEL * kv_dim) > 0);
            
            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->gqa.attn_output_weight = (float *)malloc(q_dim * D_MODEL * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->gqa.attn_output_weight, q_dim * D_MODEL) > 0);
            
            snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->gqa.attn_q_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->gqa.attn_q_norm_weight, GQA_HEAD_DIM) > 0);
            
            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->gqa.attn_k_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->gqa.attn_k_norm_weight, GQA_HEAD_DIM) > 0);
            
            if (!ok) { fprintf(stderr, "Failed to load GQA weights for layer %d\n", l); goto fail; }
            printf("  Layer %d: GQA loaded\n", l);
        }
        
        // Load MoE (FFN) weights — NOT loaded by default (memory: 3.2 GB/layer)
        // Use test_moe.c for standalone MoE testing
        layer->moe.loaded = false;
    }
    
    // Load final norm
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output_norm.weight");
    if (t) {
        model->norm_weight = (float *)malloc(D_MODEL * sizeof(float));
        gguf_read_tensor_f32(ctx, t, model->norm_weight, D_MODEL);
        printf("  Final norm loaded\n");
    } else {
        printf("  WARNING: output_norm.weight not found\n");
    }
    
    // Embeddings: auto-extract from GGUF if not available, else load from file
    model->use_embedding_file = true;
    const char *emb_path = "data/qwen36_embeddings_c.bin.raw";
    FILE *emb_f = fopen(emb_path, "rb");
    if (emb_f) {
        fseek(emb_f, 0, SEEK_END);
        long emb_size = ftell(emb_f);
        int file_vocab = (int)(emb_size / (D_MODEL * sizeof(float)));
        if (file_vocab == 248320) {
            model->vocab_size = file_vocab;
            printf("  Embeddings: %d tokens from file (%ld MB)\n", model->vocab_size, emb_size / (1024*1024));
            fclose(emb_f);
        } else {
            fclose(emb_f);
            printf("  Embedding file has wrong size (%d tokens, expected 248320), re-extracting...\n", file_vocab);
            goto extract_embeddings;
        }
    } else {
        extract_embeddings: {
            printf("  Extracting token_embd.weight from GGUF...\n");
            gguf_tensor_info *t_emb = gguf_find_tensor(ctx, "token_embd.weight");
            if (!t_emb) { fprintf(stderr, "  ERROR: token_embd.weight not found\n"); }
            else {
                int64_t n_emb = (int64_t)248320 * D_MODEL;
                float *temp_emb = (float *)malloc(n_emb * sizeof(float));
                if (temp_emb && gguf_read_tensor_f32(ctx, t_emb, temp_emb, n_emb) > 0) {
                    FILE *out = fopen(emb_path, "wb");
                    if (out) {
                        fwrite(temp_emb, sizeof(float), n_emb, out);
                        fclose(out);
                        printf("  Embeddings saved to %s\n", emb_path);
                    }
                    // Use the in-memory copy for this run
                    model->token_embd = temp_emb;
                    model->vocab_size = 248320;
                    model->use_embedding_file = false;
                    printf("  Token embeddings: %ld MB (in-memory)\n", n_emb * sizeof(float) / (1024*1024));
                } else {
                    if (temp_emb) free(temp_emb);
                    fprintf(stderr, "  Failed to extract embeddings\n");
                }
            }
        }
    }
    
    if (model->use_embedding_file) {
        // Verify vocab_size was set from file
        if (model->vocab_size == 0) model->vocab_size = 248320;
    }
    
    // Load output weight for logit projection
    gguf_tensor_info *t_out = gguf_find_tensor(ctx, "output.weight");
    if (t_out) {
        int64_t out_elems = (int64_t)D_MODEL * model->vocab_size;
        model->output_weight = (float *)malloc(out_elems * sizeof(float));
        if (gguf_read_tensor_f32(ctx, t_out, model->output_weight, out_elems) > 0)
            printf("  Output weight loaded: %ld MB\n", out_elems * sizeof(float) / (1024*1024));
        else
            { fprintf(stderr, "Failed to load output.weight\n"); free(model->output_weight); model->output_weight = NULL; }
    } else {
        printf("  WARNING: output.weight not found\n");
    }
    
    // Allocate state buffers
    int max_s = model->n_layers;
    int ssm_state_size = max_s * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    int conv_state_size = max_s * (CONV_KERNEL - 1) * CONV_DIM;
    model->ssm_states = (float *)calloc(ssm_state_size + conv_state_size, sizeof(float));
    model->conv_states = model->ssm_states + ssm_state_size;
    
    model->gguf_ctx = ctx;  // Keep ctx open for per-layer MoE loading
    model->enable_moe = false;  // MoE disabled by default (memory: 3.2 GB/layer)
    model->moe_max_layers = 0;  // 0 = all layers
    
    printf("Model initialized: %d layers (%d SSM, %d GQA), %d vocab\n",
           model->n_layers,
           model->n_layers - model->n_layers/4,
           model->n_layers/4,
           model->vocab_size);
    
    return true;
    
fail:
    gguf_close(ctx);
    model->gguf_ctx = NULL;
    wubu_model_free(model);
    return false;
}

void wubu_model_free(wubu_model_t *model) {
    if (!model) return;
    for (int l = 0; l < model->n_layers; l++) {
        wubu_layer_t *layer = &model->layers[l];
        free(layer->attn_norm_weight);
        free(layer->post_attn_norm_weight);
        // Free MoE weights
        free(layer->moe.ffn_gate_inp);
        free(layer->moe.ffn_gate_exps);
        free(layer->moe.ffn_up_exps);
        free(layer->moe.ffn_down_exps);
        free(layer->moe.ffn_gate_shexp);
        free(layer->moe.ffn_up_shexp);
        free(layer->moe.ffn_down_shexp);
        free(layer->moe.ffn_gate_inp_shexp);
        if (layer->is_ssm) {
            free(layer->ssm.attn_qkv_weight);
            free(layer->ssm.attn_gate_weight);
            free(layer->ssm.ssm_beta_weight);
            free(layer->ssm.ssm_alpha_weight);
            free(layer->ssm.ssm_dt_bias);
            free(layer->ssm.ssm_a);
            free(layer->ssm.ssm_conv1d_weight);
            free(layer->ssm.ssm_norm_weight);
            free(layer->ssm.ssm_out_weight);
        } else {
            free(layer->gqa.attn_q_weight);
            free(layer->gqa.attn_k_weight);
            free(layer->gqa.attn_v_weight);
            free(layer->gqa.attn_output_weight);
            free(layer->gqa.attn_q_norm_weight);
            free(layer->gqa.attn_k_norm_weight);
        }
    }
    free(model->layers);
    free(model->norm_weight);
    free(model->token_embd);
    free(model->output_weight);
    free(model->ssm_states);
    if (model->gguf_ctx) {
        gguf_close(model->gguf_ctx);
        model->gguf_ctx = NULL;
    }
    memset(model, 0, sizeof(*model));
}

// ========== Forward Pass ==========

void wubu_model_forward_from_embd(wubu_model_t *model,
                                  const float *embeddings, int B, int T,
                                  float *logits) {
    const int N = B * T;
    
    // Allocate residual stream
    float *x = (float *)malloc(N * D_MODEL * sizeof(float));
    memcpy(x, embeddings, N * D_MODEL * sizeof(float));
    
    // Layer loop
    for (int l = 0; l < model->n_layers; l++) {
        wubu_layer_t *layer = &model->layers[l];
        
        // Pre-attention RMSNorm
        float *normed = (float *)malloc(N * D_MODEL * sizeof(float));
        wubu_rms_norm(B, T, D_MODEL, x, layer->attn_norm_weight, 1e-6f, normed);
        
        // Attention
        float *attn_out = (float *)malloc(N * D_MODEL * sizeof(float));
        
        if (layer->is_ssm) {
            float *ssm_state = model->ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = model->conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, B, T, &layer->ssm, ssm_state, conv_state, attn_out);
        } else {
            wubu_gqa_forward(normed, B, T, &layer->gqa, attn_out);
        }
        
        // NaN check: find exact index of first NaN
        int nan_idx = -1;
        for (int i = 0; i < N * D_MODEL; i++) {
            if (isnan(attn_out[i])) { nan_idx = i; break; }
        }
        if (nan_idx >= 0) {
            int t = nan_idx / D_MODEL;
            int d = nan_idx % D_MODEL;
            printf("  L%d (%s) *** NaN at [t=%d,d=%d] val=%+.4e prev=%+.4e next=%+.4e\n",
                   l, layer->is_ssm ? "SSM" : "GQA",
                   t, d, attn_out[nan_idx],
                   nan_idx > 0 ? (double)attn_out[nan_idx-1] : 0.0,
                   nan_idx+1 < N*D_MODEL ? (double)attn_out[nan_idx+1] : 0.0);
        }
        
        // Residual: x = x + attn_out
        #pragma omp parallel for if(N * D_MODEL > 500000)
        for (int i = 0; i < N * D_MODEL; i++) x[i] += attn_out[i];
        free(attn_out);
        
        // Post-attention RMSNorm
        float *normed2 = (float *)malloc(N * D_MODEL * sizeof(float));
        wubu_rms_norm(B, T, D_MODEL, x, layer->post_attn_norm_weight, 1e-6f, normed2);
        
        // MoE (FFN) forward — per-layer lazy load if enabled
        float *ffn_out = (float *)malloc(N * D_MODEL * sizeof(float));
        if (model->enable_moe && model->gguf_ctx &&
            (model->moe_max_layers == 0 || l < model->moe_max_layers)) {
            if (wubu_moe_load_layer(model->gguf_ctx, l, &layer->moe)) {
                wubu_moe_forward(normed2, B, T, &layer->moe, ffn_out);
                wubu_moe_free_layer(&layer->moe);
            } else {
                memcpy(ffn_out, normed2, N * D_MODEL * sizeof(float));
            }
        } else {
            // Pass-through when MoE disabled
            wubu_moe_forward(normed2, B, T, &layer->moe, ffn_out);
        }
        
        // Residual: x = x + ffn_out
        #pragma omp parallel for if(N * D_MODEL > 500000)
        for (int i = 0; i < N * D_MODEL; i++) x[i] += ffn_out[i];
        
        free(normed);
        free(normed2);
        free(ffn_out);
    }
    
    // Final RMSNorm
    if (model->norm_weight) {
        float *final_normed = (float *)malloc(N * D_MODEL * sizeof(float));
        wubu_rms_norm(B, T, D_MODEL, x, model->norm_weight, 1e-6f, final_normed);
        memcpy(x, final_normed, N * D_MODEL * sizeof(float));
        free(final_normed);
    }
    
    // Output projection (into logits space)
    // logits[t, v] = sum_k h[t,k] * output_weight[k, v]
    if (model->output_weight) {
        #pragma omp parallel for collapse(2) if(N * model->vocab_size > 100000)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < model->vocab_size; j++) {
                const float *h_i = x + i * D_MODEL;
                float *log_i = logits + i * model->vocab_size;
                double sum = 0.0;
                for (int k = 0; k < D_MODEL; k++)
                    sum += (double)h_i[k] * (double)model->output_weight[j * D_MODEL + k];
                log_i[j] = (float)sum;
            }
        }
    } else {
        // Fallback: copy hidden states only (no output weight loaded)
        memcpy(logits, x, N * D_MODEL * sizeof(float));
    }
    
    free(x);
}

// ========== Backward Pass ==========

void wubu_model_backward_from_embd(
    const wubu_model_t *model,
    const float *embeddings,
    const float *logits, const float *d_logits,
    const float *saved_normed,     // [n_layers * N * D_MODEL]
    const float *saved_attn_out,   // [n_layers * N * D_MODEL]
    const float *saved_normed2,    // [n_layers * N * D_MODEL]
    const float *saved_ffn_out,    // [n_layers * N * D_MODEL]
    float *d_embeddings,
    int B, int T)
{
    const int N = B * T;
    const int n_layers = model->n_layers;
    const int layer_sz = N * D_MODEL;
    
    float *d_x = (float *)malloc(N * D_MODEL * sizeof(float));
    memcpy(d_x, d_logits, N * D_MODEL * sizeof(float));
    
    // Per-layer temp state buffers (reused via ssm_states/conv_states in model)
    // For exact backward, we need to re-run the forward with save
    
    // Process layers in reverse
    for (int l = n_layers - 1; l >= 0; l--) {
        const wubu_layer_t *layer = &model->layers[l];
        const float *normed = saved_normed + l * layer_sz;
        const float *attn_out = saved_attn_out + l * layer_sz;
        const float *normed2 = saved_normed2 + l * layer_sz;
        
        float *d_ffn_out = (float *)malloc(N * D_MODEL * sizeof(float));
        float *d_x_after_attn = (float *)malloc(N * D_MODEL * sizeof(float));
        float *d_attn_out = (float *)malloc(N * D_MODEL * sizeof(float));
        memcpy(d_ffn_out, d_x, layer_sz);
        memcpy(d_x_after_attn, d_x, layer_sz);
        
        // Post-attention RMSNorm backward
        wubu_rms_norm_backward(B, T, D_MODEL, normed2, layer->post_attn_norm_weight,
                               1e-6f, d_ffn_out, d_x_after_attn);
        memcpy(d_attn_out, d_x_after_attn, layer_sz);
        
        // Layer backward — exact with saved intermediates
        float *d_normed = (float *)calloc(N * D_MODEL, sizeof(float));
        
        if (layer->is_ssm) {
            // Re-run SSM forward WITH save to capture intermediates for backward
            ssm_fwd_save_t save;
            memset(&save, 0, sizeof(save));
            
            // Allocate save buffers for one layer
            float *ssm_state_tmp = model->ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state_tmp = model->conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            
            int state_sz = SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            
            // We need a separate states_t buffer (not the in-place one)
            float *states_t = (float *)malloc((T+1) * state_sz * sizeof(float));
            float *qkv_all_b = (float *)malloc(N * CONV_DIM * sizeof(float));
            float *z_all_b = (float *)malloc(N * VALUE_DIM * sizeof(float));
            float *beta_raw_b = (float *)malloc(N * DT_RANK * sizeof(float));
            float *alpha_raw_b = (float *)malloc(N * DT_RANK * sizeof(float));
            float *conv_out_b = (float *)malloc(N * CONV_DIM * sizeof(float));
            float *q_conv_b = (float *)malloc(N * KEY_DIM * sizeof(float));
            float *k_conv_b = (float *)malloc(N * KEY_DIM * sizeof(float));
            float *v_conv_b = (float *)malloc(N * VALUE_DIM * sizeof(float));
            float *q_norm_b = (float *)malloc(N * KEY_DIM * sizeof(float));
            float *k_norm_b = (float *)malloc(N * KEY_DIM * sizeof(float));
            float *delta_out_b = (float *)malloc(N * VALUE_DIM * sizeof(float));
            float *z_silu_b = (float *)malloc(N * VALUE_DIM * sizeof(float));
            float *beta_flat_b = (float *)malloc(N * DT_RANK * sizeof(float));
            float *gate_flat_b = (float *)malloc(N * DT_RANK * sizeof(float));
            float *conv_state_copy = (float *)malloc((CONV_KERNEL-1) * CONV_DIM * sizeof(float));
            
            if (!states_t || !qkv_all_b || !z_all_b || !beta_raw_b || !alpha_raw_b ||
                !conv_out_b || !q_conv_b || !k_conv_b || !v_conv_b ||
                !q_norm_b || !k_norm_b || !delta_out_b || !z_silu_b ||
                !beta_flat_b || !gate_flat_b || !conv_state_copy) {
                fprintf(stderr, "model backward SSM save alloc failed\n");
                free(states_t); free(qkv_all_b); free(z_all_b);
                free(beta_raw_b); free(alpha_raw_b);
                free(conv_out_b); free(q_conv_b); free(k_conv_b); free(v_conv_b);
                free(q_norm_b); free(k_norm_b); free(delta_out_b); free(z_silu_b);
                free(beta_flat_b); free(gate_flat_b); free(conv_state_copy);
                free(d_ffn_out); free(d_x_after_attn); free(d_attn_out); free(d_normed);
                free(d_x); return;
            }
            
            save.states_t = states_t;
            save.qkv_all = qkv_all_b;
            save.z_all = z_all_b;
            save.beta_raw = beta_raw_b;
            save.alpha_raw = alpha_raw_b;
            save.conv_post_silu = conv_out_b;
            save.q_conv = q_conv_b;
            save.k_conv = k_conv_b;
            save.v_conv = v_conv_b;
            save.q_norm = q_norm_b;
            save.k_norm = k_norm_b;
            save.delta_out = delta_out_b;
            save.z_silu = z_silu_b;
            save.beta_flat = beta_flat_b;
            save.gate_flat = gate_flat_b;
            save.conv_state_copy = conv_state_copy;
            
            // Save current SSM state, run save-forward, then restore
            float *saved_ssm_state = (float *)malloc(state_sz * sizeof(float));
            memcpy(saved_ssm_state, ssm_state_tmp, state_sz * sizeof(float));
            
            // Run forward with save — attn_out goes to a dummy buffer
            float *fwd_out = (float *)malloc(N * D_MODEL * sizeof(float));
            wubu_ssm_forward_save(normed, B, T, &layer->ssm,
                                   ssm_state_tmp, conv_state_tmp,
                                   fwd_out, &save);
            
            // Run exact backward
            wubu_ssm_backward(B, T, normed, attn_out, d_attn_out,
                              save.qkv_all, save.z_all,
                              save.beta_raw, save.alpha_raw,
                              save.conv_post_silu,
                              save.q_conv, save.k_conv, save.v_conv,
                              save.q_norm, save.k_norm,
                              save.delta_out, save.z_silu,
                              save.states_t,
                              save.beta_flat, save.gate_flat,
                              save.conv_state_copy,
                              &layer->ssm,
                              d_normed, NULL, NULL, NULL, NULL,
                              NULL, NULL, NULL, NULL);
            
            // Restore SSM state
            memcpy(ssm_state_tmp, saved_ssm_state, state_sz * sizeof(float));
            
            free(saved_ssm_state);
            free(fwd_out);
            free(states_t); free(qkv_all_b); free(z_all_b);
            free(beta_raw_b); free(alpha_raw_b);
            free(conv_out_b); free(q_conv_b); free(k_conv_b); free(v_conv_b);
            free(q_norm_b); free(k_norm_b); free(delta_out_b); free(z_silu_b);
            free(beta_flat_b); free(gate_flat_b); free(conv_state_copy);
            
        } else {
            // GQA backward with saved intermediates
            gqa_fwd_save_t save;
            memset(&save, 0, sizeof(save));
            
            int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
            int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
            
            float *Q_norm_b = (float *)malloc(N * q_dim * sizeof(float));
            float *Q_raw_b = (float *)malloc(N * q_dim * sizeof(float));
            float *K_norm_b = (float *)malloc(N * kv_dim * sizeof(float));
            float *K_raw_b = (float *)malloc(N * kv_dim * sizeof(float));
            float *V_b = (float *)malloc(N * kv_dim * sizeof(float));
            float *gate_b = (float *)malloc(N * q_dim * sizeof(float));
            float *gate_sig_b = (float *)malloc(N * q_dim * sizeof(float));
            float *attn_pre_gate_b = (float *)malloc(N * q_dim * sizeof(float));
            
            if (!Q_norm_b || !Q_raw_b || !K_norm_b || !K_raw_b || !V_b ||
                !gate_b || !gate_sig_b || !attn_pre_gate_b) {
                fprintf(stderr, "model backward GQA save alloc failed\n");
                free(Q_norm_b); free(Q_raw_b); free(K_norm_b); free(K_raw_b);
                free(V_b); free(gate_b); free(gate_sig_b); free(attn_pre_gate_b);
                free(d_ffn_out); free(d_x_after_attn); free(d_attn_out); free(d_normed);
                free(d_x); return;
            }
            
            save.Q_norm = Q_norm_b;
            save.Q_raw = Q_raw_b;
            save.K_norm = K_norm_b;
            save.K_raw = K_raw_b;
            save.V = V_b;
            save.gate = gate_b;
            save.gate_sig = gate_sig_b;
            save.attn_out_pre_gate = attn_pre_gate_b;
            
            // Run forward with save
            float *fwd_out = (float *)malloc(N * D_MODEL * sizeof(float));
            wubu_gqa_forward_save(normed, B, T, &layer->gqa, fwd_out, &save);
            
            // Run exact backward
            wubu_gqa_backward(B, T, normed,
                              save.Q_norm, save.Q_raw,
                              save.K_norm, save.K_raw,
                              save.V,
                              save.gate, save.gate_sig,
                              save.attn_out_pre_gate, attn_out,
                              d_attn_out,
                              &layer->gqa,
                              d_normed,
                              NULL, NULL, NULL, NULL, NULL, NULL);
            
            free(fwd_out);
            free(Q_norm_b); free(Q_raw_b); free(K_norm_b); free(K_raw_b);
            free(V_b); free(gate_b); free(gate_sig_b); free(attn_pre_gate_b);
        }
        
        // Pre-attention RMSNorm backward
        float *d_x_pre_attn = (float *)malloc(N * D_MODEL * sizeof(float));
        memset(d_x_pre_attn, 0, layer_sz);
        wubu_rms_norm_backward(B, T, D_MODEL, normed, layer->attn_norm_weight,
                               1e-6f, d_normed, d_x_pre_attn);
        
        // Residual: x_pre_attn also feeds x_after_attn = x_pre_attn + attn_out
        for (int i = 0; i < N * D_MODEL; i++)
            d_x_pre_attn[i] += d_x_after_attn[i];
        
        memcpy(d_x, d_x_pre_attn, layer_sz);
        
        free(d_ffn_out);
        free(d_x_after_attn);
        free(d_attn_out);
        free(d_normed);
        free(d_x_pre_attn);
    }
    
    memcpy(d_embeddings, d_x, N * D_MODEL * sizeof(float));
    free(d_x);
}

void wubu_model_forward(wubu_model_t *model,
                        const int *token_ids, int B, int T,
                        float *logits) {
    // Load embeddings
    float *embd = (float *)malloc(B * T * D_MODEL * sizeof(float));
    
    if (model->token_embd && !model->use_embedding_file) {
        // Use GGUF-loaded embeddings (auto-extracted)
        for (int i = 0; i < B * T; i++) {
            int id = token_ids[i];
            if (id < 0 || id >= model->vocab_size) id = 0;
            memcpy(embd + i * D_MODEL, model->token_embd + id * D_MODEL, D_MODEL * sizeof(float));
        }
    } else if (model->use_embedding_file) {
        FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (f) {
            for (int i = 0; i < B * T; i++) {
                int id = token_ids[i];
                if (id < 0 || id >= model->vocab_size) id = 0;
                fseek(f, id * D_MODEL * sizeof(float), SEEK_SET);
                fread(embd + i * D_MODEL, sizeof(float), D_MODEL, f);
            }
            fclose(f);
        } else {
            memset(embd, 0, B * T * D_MODEL * sizeof(float));
        }
    } else {
        memset(embd, 0, B * T * D_MODEL * sizeof(float));
    }
    
    wubu_model_forward_from_embd(model, embd, B, T, logits);
    free(embd);
}
