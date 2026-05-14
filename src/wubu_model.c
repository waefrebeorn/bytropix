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
    
    // Embeddings: use pre-extracted file from Phase 1
    model->use_embedding_file = true;
    FILE *emb_f = fopen("data/qwen36_embeddings_c.bin", "rb");
    if (emb_f) {
        fseek(emb_f, 0, SEEK_END);
        long emb_size = ftell(emb_f);
        fseek(emb_f, 0, SEEK_SET);
        model->vocab_size = (int)(emb_size / (D_MODEL * sizeof(float)));
        printf("  Embeddings: %d tokens from file (%ld MB)\n", model->vocab_size, emb_size / (1024*1024));
        fclose(emb_f);
    }
    
    // Allocate state buffers
    int max_s = model->n_layers;
    int ssm_state_size = max_s * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    int conv_state_size = max_s * (CONV_KERNEL - 1) * CONV_DIM;
    model->ssm_states = (float *)calloc(ssm_state_size + conv_state_size, sizeof(float));
    model->conv_states = model->ssm_states + ssm_state_size;
    
    printf("Model initialized: %d layers (%d SSM, %d GQA), %d vocab\n",
           model->n_layers,
           model->n_layers - model->n_layers/4,
           model->n_layers/4,
           model->vocab_size);
    
    gguf_close(ctx);
    return true;
    
fail:
    gguf_close(ctx);
    wubu_model_free(model);
    return false;
}

void wubu_model_free(wubu_model_t *model) {
    if (!model) return;
    for (int l = 0; l < model->n_layers; l++) {
        wubu_layer_t *layer = &model->layers[l];
        free(layer->attn_norm_weight);
        free(layer->post_attn_norm_weight);
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
    free(model->ssm_states);
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
        for (int i = 0; i < N * D_MODEL; i++) x[i] += attn_out[i];
        free(attn_out);
        
        // Post-attention RMSNorm
        float *normed2 = (float *)malloc(N * D_MODEL * sizeof(float));
        wubu_rms_norm(B, T, D_MODEL, x, layer->post_attn_norm_weight, 1e-6f, normed2);
        
        // FFN placeholder: just pass through (Phase 4 will add MoE)
        memcpy(x, normed2, N * D_MODEL * sizeof(float));
        
        free(normed);
        free(normed2);
    }
    
    // Final RMSNorm
    if (model->norm_weight) {
        float *final_normed = (float *)malloc(N * D_MODEL * sizeof(float));
        wubu_rms_norm(B, T, D_MODEL, x, model->norm_weight, 1e-6f, final_normed);
        memcpy(x, final_normed, N * D_MODEL * sizeof(float));
        free(final_normed);
    }
    
    // Output projection (into logits space)
    // No output weight loaded yet — just copy x to logits for now
    memcpy(logits, x, N * D_MODEL * sizeof(float));
    
    free(x);
}

void wubu_model_forward(wubu_model_t *model,
                        const int *token_ids, int B, int T,
                        float *logits) {
    // Load embeddings from file
    float *embd = (float *)malloc(B * T * D_MODEL * sizeof(float));
    
    if (model->use_embedding_file) {
        FILE *f = fopen("data/qwen36_embeddings_c.bin", "rb");
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
