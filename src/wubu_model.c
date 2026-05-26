#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>  // _mm_prefetch for expert prefetch

#include <omp.h>

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
    int has_nextn = 0;
    for (int i = 0; i < (int)ctx->n_tensors; i++) {
        const char *name = ctx->tensors[i].name;
        if (strncmp(name, "blk.", 4) == 0) {
            int layer = atoi(name + 4);
            if (layer > max_layer) max_layer = layer;
            // Check if this is an MTP model (has nextn.* tensors)
            if (strstr(name, ".nextn.")) has_nextn = 1;
        }
    }
    // For MTP models, the last layer (blk.40) is the MTP prediction head
    // Only count regular layers (skip MTP head)
    if (has_nextn) {
        model->n_layers = max_layer;  // 40 layers (0..39) for MTP model
        printf("MTP model detected: %d regular layers + 1 MTP head\n", max_layer);
    } else {
        model->n_layers = max_layer + 1;  // 41 layers for MTP, 40 for regular
    }
    
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
            // Load SSM weights — QUANTIZED-ONLY PATH for large weight matrices.
            // attn_qkv, attn_gate, ssm_out use quantized blob pointers (set later).
            // Small tensors (norms, a, dt, conv1d) loaded as F32.
            char name[256];
            int ok = 1;
            
            // LARGE: attn_qkv_weight — quantized-only (blob pointer)
            layer->ssm.attn_qkv_weight = NULL;
            snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            
            // LARGE: attn_gate_weight — quantized-only (blob pointer)
            layer->ssm.attn_gate_weight = NULL;
            snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            
            // Small: ssm_beta.weight [2048, 32] F32
            snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->ssm.ssm_beta_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_beta_weight, D_MODEL * DT_RANK) > 0);
            
            // Small: ssm_alpha.weight [2048, 32] F32
            snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->ssm.ssm_alpha_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_alpha_weight, D_MODEL * DT_RANK) > 0);
            
            // Small: ssm_dt.bias [32] F32
            snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->ssm.ssm_dt_bias = (float *)malloc(DT_RANK * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_dt_bias, DT_RANK) > 0);
            
            // Small: ssm_a [32] F32
            snprintf(name, sizeof(name), "blk.%d.ssm_a", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->ssm.ssm_a = (float *)malloc(DT_RANK * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_a, DT_RANK) > 0);
            
            // Small: ssm_conv1d.weight [4, 8192] = 128KB F32
            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->ssm.ssm_conv1d_weight = (float *)malloc(CONV_KERNEL * CONV_DIM * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_conv1d_weight, CONV_KERNEL * CONV_DIM) > 0);
            
            // Small: ssm_norm.weight [128] F32
            snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->ssm.ssm_norm_weight = (float *)malloc(SSM_D_STATE * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_norm_weight, SSM_D_STATE) > 0);
            
            // LARGE: ssm_out.weight — quantized-only (blob pointer)
            layer->ssm.ssm_out_weight = NULL;
            
            if (!ok) { fprintf(stderr, "Failed to load SSM weights for layer %d\n", l); goto fail; }
            printf("  Layer %d: SSM loaded (quantized attn_qkv/gate/out)\n", l);
            
        } else {
            // Load GQA weights — QUANTIZED-ONLY PATH for large weight matrices.
            // attn_q, attn_k, attn_v, attn_output use quantized blob pointers (set later).
            char name[256];
            int ok = 1;
            
            // LARGE: attn_q.weight — quantized-only (blob pointer)
            layer->gqa.attn_q_weight = NULL;
            
            // LARGE: attn_k.weight — quantized-only (blob pointer)
            layer->gqa.attn_k_weight = NULL;
            
            // LARGE: attn_v.weight — quantized-only (blob pointer)
            layer->gqa.attn_v_weight = NULL;
            
            // LARGE: attn_output.weight — quantized-only (blob pointer)
            layer->gqa.attn_output_weight = NULL;
            
            // Small: attn_q_norm.weight [256] F32
            snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->gqa.attn_q_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->gqa.attn_q_norm_weight, GQA_HEAD_DIM) > 0);
            
            // Small: attn_k_norm.weight [256] F32
            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            layer->gqa.attn_k_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
            ok = ok && (gguf_read_tensor_f32(ctx, t, layer->gqa.attn_k_norm_weight, GQA_HEAD_DIM) > 0);
            
            if (!ok) { fprintf(stderr, "Failed to load GQA weights for layer %d\n", l); goto fail; }
            printf("  Layer %d: GQA loaded (quantized attn_q/k/v/output)\n", l);
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
    
    // Load output weight for logit projection — QUANTIZED-ONLY (Q4_K blob pointer)
    model->output_weight = NULL;
    gguf_tensor_info *t_out = gguf_find_tensor(ctx, "output.weight");
    if (!t_out) { fprintf(stderr, "  ERROR: output.weight not found\n"); }
    
    // Output weight quantized pointer will be set after gguf_buffer_data() below
    printf("  Output weight: will use quantized path (Q4_K via blob pointer)\n");
    
    // Allocate state buffers
    int max_s = model->n_layers;
    int ssm_state_size = max_s * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    int conv_state_size = max_s * (CONV_KERNEL - 1) * CONV_DIM;
    model->ssm_states = (float *)calloc(ssm_state_size + conv_state_size, sizeof(float));
    model->conv_states = model->ssm_states + ssm_state_size;
    
    model->gguf_ctx = ctx;  // Keep ctx open for per-layer MoE loading
    model->enable_moe = false;  // MoE disabled by default (memory: 3.2 GB/layer)
    model->moe_max_layers = 0;  // 0 = all layers
    
    // Read SSM L2 norm epsilon from GGUF config (qwen35moe.attention.layer_norm_rms_epsilon = 1e-6)
    g_ssm_l2_eps = 1e-6f;
    printf("  SSM L2 eps: %e\n", g_ssm_l2_eps);
    
    // Buffer GGUF data and save quantized weight pointers
    gguf_buffer_data(ctx);
    {
        const uint8_t *blob = (const uint8_t *)ctx->data_blob;
        for (int l = 0; l < model->n_layers; l++) {
            wubu_layer_t *layer = &model->layers[l];
            gguf_tensor_info *t;
            char name[256];
            if (layer->is_ssm) {
                snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", l);
                t = gguf_find_tensor(ctx, name);
                if (t && blob) { layer->ssm.attn_qkv_weight_q = blob + t->data_offset; layer->ssm.attn_qkv_weight_type = t->ggml_type; }
                snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", l);
                t = gguf_find_tensor(ctx, name);
                if (t && blob) { layer->ssm.attn_gate_weight_q = blob + t->data_offset; layer->ssm.attn_gate_weight_type = t->ggml_type; }
                snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", l);
                t = gguf_find_tensor(ctx, name);
                if (t && blob) { layer->ssm.ssm_out_weight_q = blob + t->data_offset; layer->ssm.ssm_out_weight_type = t->ggml_type; }
            } else {
                snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
                t = gguf_find_tensor(ctx, name);
                if (t && blob) { layer->gqa.attn_q_weight_q = blob + t->data_offset; layer->gqa.attn_q_weight_type = t->ggml_type; }
                snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
                t = gguf_find_tensor(ctx, name);
                if (t && blob) { layer->gqa.attn_k_weight_q = blob + t->data_offset; layer->gqa.attn_k_weight_type = t->ggml_type; }
                snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
                t = gguf_find_tensor(ctx, name);
                if (t && blob) { layer->gqa.attn_v_weight_q = blob + t->data_offset; layer->gqa.attn_v_weight_type = t->ggml_type; }
                snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
                t = gguf_find_tensor(ctx, name);
                if (t && blob) { layer->gqa.attn_output_weight_q = blob + t->data_offset; layer->gqa.attn_output_weight_type = t->ggml_type; }
            }
        }
        gguf_tensor_info *t_out = gguf_find_tensor(ctx, "output.weight");
        if (t_out && blob) { model->output_weight_q = blob + t_out->data_offset; model->output_weight_type = t_out->ggml_type; }

        // Save MoE quantized pointers for each layer (routed + shared experts)
        for (int l = 0; l < model->n_layers; l++) {
            wubu_layer_t *layer = &model->layers[l];
            gguf_tensor_info *t;
            char name[256];
            moe_weights_t *moe = &layer->moe;

            // Router is F32 — direct pointer from blob
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (t && blob) { moe->ffn_gate_inp = (float *)(blob + t->data_offset); }

            // Shared expert gate weight (F32)
            snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp_shexp.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (t && blob) { moe->ffn_gate_inp_shexp = (float *)(blob + t->data_offset); }

            snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (t && blob) { moe->ffn_gate_exps_q = blob + t->data_offset; moe->ffn_gate_exps_q_type = t->ggml_type; }

            snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (t && blob) { moe->ffn_up_exps_q = blob + t->data_offset; moe->ffn_up_exps_q_type = t->ggml_type; }

            snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (t && blob) { moe->ffn_down_exps_q = blob + t->data_offset; moe->ffn_down_exps_q_type = t->ggml_type; }

            snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (t && blob) { moe->ffn_gate_shexp_q = blob + t->data_offset; moe->ffn_gate_shexp_q_type = t->ggml_type; }

            snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (t && blob) { moe->ffn_up_shexp_q = blob + t->data_offset; moe->ffn_up_shexp_q_type = t->ggml_type; }

            snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (t && blob) { moe->ffn_down_shexp_q = blob + t->data_offset; moe->ffn_down_shexp_q_type = t->ggml_type; }

            // Mark MoE as loaded for quantized path
            if (moe->ffn_gate_exps_q && moe->ffn_up_exps_q && moe->ffn_down_exps_q) {
                moe->loaded = true;
                moe->load_from_blob = true;
            }
        }
    }
    
    printf("Model initialized: %d layers (%d SSM, %d GQA), %d vocab\n",
           model->n_layers,
           model->n_layers - model->n_layers/4,
           model->n_layers/4,
           model->vocab_size);
    
    // Allocate GQA KV cache (10 GQA layers × 256k context × 512 dim)
    int64_t cache_elems = (int64_t)10 * GQA_MAX_CTX * GQA_KV_DIM;
    model->gqa_k_cache = malloc(kv_cache_alloc_size(cache_elems));
    model->gqa_v_cache = malloc(kv_cache_alloc_size(cache_elems));
    memset(model->gqa_k_cache, 0, kv_cache_alloc_size(cache_elems));
    memset(model->gqa_v_cache, 0, kv_cache_alloc_size(cache_elems));
    model->gqa_cache_len = 0;
    
    return true;
    
fail:
    gguf_close(ctx);
    model->gguf_ctx = NULL;
    wubu_model_free(model);
    return false;
}

void wubu_model_free(wubu_model_t *model) {
    if (!model) return;
    // Free GPU resources first
#ifdef GPU_SUPPORT
    wubu_model_gpu_free(model);
#endif
    for (int l = 0; l < model->n_layers; l++) {
        wubu_layer_t *layer = &model->layers[l];
        free(layer->attn_norm_weight);
        free(layer->post_attn_norm_weight);
        // Free MoE weights (skip if blob-backed)
        if (!layer->moe.load_from_blob) {
            free(layer->moe.ffn_gate_inp);
            free(layer->moe.ffn_gate_exps);
            free(layer->moe.ffn_up_exps);
            free(layer->moe.ffn_down_exps);
            free(layer->moe.ffn_gate_shexp);
            free(layer->moe.ffn_up_shexp);
            free(layer->moe.ffn_down_shexp);
            free(layer->moe.ffn_gate_inp_shexp);
        }
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
    free(model->ssm_states_saved);  // frees both ssm_states_saved and conv_states_saved (same alloc)
    free(model->gqa_k_cache);
    free(model->gqa_v_cache);
    wubu_mtp_free(&model->mtp);
    if (model->gguf_ctx) {
        gguf_close(model->gguf_ctx);
        model->gguf_ctx = NULL;
    }
    memset(model, 0, sizeof(*model));
}

static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ========== Forward Pass ==========

void wubu_model_forward_from_embd(wubu_model_t *model,
                                  const float *embeddings, int B, int T,
                                  float *logits) {
    const int N = B * T;
    
    // Allocate residual stream + reusable buffers (avoids 160 mallocs per forward)
    float *x = (float *)malloc(N * D_MODEL * sizeof(float));
    memcpy(x, embeddings, N * D_MODEL * sizeof(float));
    float *normed = (float *)malloc(N * D_MODEL * sizeof(float));
    float *attn_out = (float *)malloc(N * D_MODEL * sizeof(float));
    float *normed2 = (float *)malloc(N * D_MODEL * sizeof(float));
    float *ffn_out = (float *)malloc(N * D_MODEL * sizeof(float));
    int *prev_experts = (int *)malloc(N * N_ACTIVE_EXPTS * sizeof(int));
    int have_prev_experts = 0;
    
    // Layer loop
    for (int l = 0; l < model->n_layers; l++) {
        wubu_layer_t *layer = &model->layers[l];
        
        // DEBUG: dump hidden after each layer
        static int dump_layer = -1;
        const char *dl_env = getenv("DUMP_LAYER");
        if (dl_env) dump_layer = atoi(dl_env);
        if (l == dump_layer) {
            FILE *f = fopen("/tmp/debug_hidden_before_l.bin", "wb");
            if (f) { fwrite(x, sizeof(float), N * D_MODEL, f); fclose(f); }
        }
        
        // Pre-attention RMSNorm
        wubu_rms_norm(B, T, D_MODEL, x, layer->attn_norm_weight, 1e-6f, normed);
        
        // Expert prefetch: if previous layer had MoE, prefetch this layer's expert weights
        // Uses the previous layer's selected expert indices (experts tend to persist across layers)
        // Strides through full weight data to L3 cache, not just first 256 bytes to L1
        if (have_prev_experts && l > 0 && layer->moe.loaded && layer->moe.ffn_gate_exps_q) {
            wubu_layer_t *prev = &model->layers[l-1];
            if (prev->moe.loaded) {
                int64_t gate_bytes = gguf_raw_size(layer->moe.ffn_gate_exps_q_type, (int64_t)D_MODEL * D_FF);
                int64_t up_bytes   = gguf_raw_size(layer->moe.ffn_up_exps_q_type,   (int64_t)D_MODEL * D_FF);
                int64_t down_bytes = gguf_raw_size(layer->moe.ffn_down_exps_q_type, (int64_t)D_FF * D_MODEL);
                const int64_t P_STRIDE = 256;  // 4 cache lines per prefetch
                for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                    int e = prev_experts[k];
                    if (e < 0 || e >= N_EXPERTS) continue;
                    const uint8_t *g = layer->moe.ffn_gate_exps_q + (int64_t)e * gate_bytes;
                    const uint8_t *u = layer->moe.ffn_up_exps_q   + (int64_t)e * up_bytes;
                    const uint8_t *d = layer->moe.ffn_down_exps_q + (int64_t)e * down_bytes;
                    // Stride through full weight: ~264KB per gate/up, ~392KB per down
                    // Total ~920KB per expert, 7.4MB for 8 experts → L3
                    for (int64_t off = 0; off < gate_bytes; off += P_STRIDE) {
                        _mm_prefetch((const char *)g + off, _MM_HINT_T2);
                    }
                    for (int64_t off = 0; off < up_bytes; off += P_STRIDE) {
                        _mm_prefetch((const char *)u + off, _MM_HINT_T2);
                    }
                    for (int64_t off = 0; off < down_bytes; off += P_STRIDE) {
                        _mm_prefetch((const char *)d + off, _MM_HINT_T2);
                    }
                }
            }
        }
        
        double t0 = wall_time();
        
        if (layer->is_ssm) {
            float *ssm_state = model->ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = model->conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
#ifdef GPU_SUPPORT
            if (model->gpu_ctx && N > 1) {
                // Full GPU SSM forward for prefill (N>1): avoids per-token H2D/D2H
                int gpu_ok = wubu_model_gpu_ssm_forward_full(model, l, normed, N, attn_out);
                if (!gpu_ok) {
                    // Fallback: GPU projections + CPU conv/norm/recurrence
                    float *gpu_qkv = (float*)malloc(sizeof(float) * N * CONV_DIM);
                    float *gpu_z = (float*)malloc(sizeof(float) * N * VALUE_DIM);
                    int alloc_ok = (gpu_qkv && gpu_z);
                    if (alloc_ok) {
                        // Batched SSM projection: all N tokens at once (avoids N*H2D/D2H overhead)
                        wubu_model_gpu_ssm_project(model, l,
                            normed, N,
                            gpu_qkv, gpu_z, NULL);
                        // Set GPU recurrence pointers so wubu_ssm_forward uses GPU
                        wubu_gpu_set_ssm_hybrid(model->gpu_ctx, l, &layer->ssm);
                        wubu_ssm_forward(normed, B, T, &layer->ssm,
                            ssm_state, conv_state, attn_out, gpu_qkv, gpu_z);
                        // Clear GPU pointers to avoid stale state
                        layer->ssm.gpu_ssm_state = NULL;
                        layer->ssm.gpu_stream    = NULL;
                    } else {
                        // Allocation failed, fall back to CPU
                        wubu_ssm_forward(normed, B, T, &layer->ssm,
                            ssm_state, conv_state, attn_out, NULL, NULL);
                    }
                    free(gpu_qkv);
                    free(gpu_z);
                }
                // Sync CPU→GPU state after hybrid prefill, so forward_full
                // decode uses correct accumulated state for subsequent tokens
                if (gpu_ok) {
                    // forward_full succeeded — GPU state already correct, no sync needed
                } else {
                    wubu_gpu_sync_ssm_state_to_gpu(model->gpu_ctx, l,
                        ssm_state, conv_state);
                }
            } else if (model->gpu_ctx) {
                // N==1 decode path: use HYBRID ONLY (GPU quant matmuls + CPU SSM)
                // Sync CPU state to GPU for hybrid recurrence
                wubu_gpu_sync_ssm_state_to_gpu(model->gpu_ctx, l,
                    ssm_state, conv_state);
                wubu_gpu_set_ssm_hybrid(model->gpu_ctx, l, &layer->ssm);
                wubu_ssm_forward(normed, B, T, &layer->ssm,
                    ssm_state, conv_state, attn_out, NULL, NULL);
                layer->ssm.gpu_ssm_state = NULL;
                layer->ssm.gpu_stream    = NULL;
            } else
#endif
            {
                wubu_ssm_forward(normed, B, T, &layer->ssm,
                    ssm_state, conv_state, attn_out, NULL, NULL);
            }
        } else {
#ifdef GPU_SUPPORT
            if (model->gpu_ctx) {
                // Use cached GQA layer index to check if GPU attention is beneficial
                int gqa_use_gpu = 0;
                if (N > 1) gqa_use_gpu = 1;
                if (gqa_use_gpu) {
                    // Batched GQA: process all tokens at once (avoids N*H2D/D2H overhead)
                    int chunk_sz = wubu_model_gpu_chunk_sz(model);
                    if (N <= chunk_sz) {
                        wubu_model_gpu_gqa_forward(model, l, normed, N, attn_out);
                    } else {
                        // N exceeds GPU scratch chunk size — process in sub-batches
                        int remaining = N, offset = 0;
                        while (remaining > 0) {
                            int c = remaining < chunk_sz ? remaining : chunk_sz;
                            wubu_model_gpu_gqa_forward(model, l,
                                normed + offset * D_MODEL, c,
                                attn_out + offset * D_MODEL);
                            offset += c;
                            remaining -= c;
                        }
                    }
                    goto gqa_done;
                }
            }
#endif
            {  // CPU GQA forward with KV cache
            int l_gqa = 0;  // GQA layer index among GQA layers
            // Count GQA layers up to current to index into cache
            for (int li = 0; li < l; li++) {
                if (!model->layers[li].is_ssm) l_gqa++;
            }
            int64_t layer_cache_off = (int64_t)l_gqa * GQA_MAX_CTX * GQA_KV_DIM;
            void *k_cache = (uint8_t *)model->gqa_k_cache + kv_cache_alloc_size(layer_cache_off);
            void *v_cache = (uint8_t *)model->gqa_v_cache + kv_cache_alloc_size(layer_cache_off);
            void *k_out = (model->gqa_cache_len > 0) ? 
                ((uint8_t *)k_cache + kv_cache_alloc_size((int64_t)model->gqa_cache_len * GQA_KV_DIM)) : NULL;
            void *v_out = (model->gqa_cache_len > 0) ?
                ((uint8_t *)v_cache + kv_cache_alloc_size((int64_t)model->gqa_cache_len * GQA_KV_DIM)) : NULL;
            const void *k_in = (model->gqa_cache_len > 0) ? k_cache : NULL;
            const void *v_in = (model->gqa_cache_len > 0) ? v_cache : NULL;
            // For prefill (T>1 and first call): store to cache position 0
            if (T > 1 && model->gqa_cache_len == 0) {
                k_out = k_cache;
                v_out = v_cache;
                k_in = NULL; v_in = NULL;
            }
            wubu_gqa_forward(normed, B, T, &layer->gqa, attn_out,
                             k_in, v_in, model->gqa_cache_len,
                             k_out, v_out);
            }  // close CPU GQA block
        gqa_done:
        }  // close else block (non-SSM)
        
        double t1 = wall_time();
        if (getenv("PROFILE") && l < 3) {
            fprintf(stderr, "  L%d %s attn: %.3fms\n", l, layer->is_ssm ? "SSM" : "GQA", (t1 - t0) * 1000.0);
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
        
        // Post-attention RMSNorm
        wubu_rms_norm(B, T, D_MODEL, x, layer->post_attn_norm_weight, 1e-6f, normed2);
        
        // MoE (FFN) forward — use quantized path when available
        double t_moe0 = wall_time();
        if (layer->moe.loaded && model->enable_moe &&
            (model->moe_max_layers == 0 || l < model->moe_max_layers)) {
            // Quantized path: also save selected expert indices for next-layer prefetch
            // GPU MoE (disabled by FORCE_CPU_MOE env var for debug)
#ifdef GPU_SUPPORT
            if (model->gpu_ctx && !getenv("FORCE_CPU_MOE")) {
                layer->moe.gpu_ctx = (void *)model;
            }
#endif
            wubu_moe_forward(normed2, B, T, &layer->moe, ffn_out, have_prev_experts ? prev_experts : NULL);
            have_prev_experts = 1;
#ifdef GPU_SUPPORT
            layer->moe.gpu_ctx = NULL;  // reset after use
#endif
        } else if (model->enable_moe && model->gguf_ctx &&
                   (model->moe_max_layers == 0 || l < model->moe_max_layers)) {
            // Fallback: F32 dequant path
            if (wubu_moe_load_layer(model->gguf_ctx, l, &layer->moe)) {
                wubu_moe_forward(normed2, B, T, &layer->moe, ffn_out, NULL);
                wubu_moe_free_layer(&layer->moe);
            } else {
                memcpy(ffn_out, normed2, N * D_MODEL * sizeof(float));
            }
        } else {
            // Pass-through when MoE disabled
            memcpy(ffn_out, normed2, N * D_MODEL * sizeof(float));
        }
        
        double t_moe1 = wall_time();
        if (getenv("PROFILE") && l < 3) {
            fprintf(stderr, "  L%d MoE: %.3fms\n", l, (t_moe1 - t_moe0) * 1000.0);
        }
        
        // Residual: x = x + ffn_out
        #pragma omp parallel for if(N * D_MODEL > 500000)
        for (int i = 0; i < N * D_MODEL; i++) x[i] += ffn_out[i];
        
        // Dump per-layer hidden state (post-MoE residual = next layer's input)
        const char *dump_dir = getenv("DUMP_LAYER_DIR");
        if (dump_dir) {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s/our_layer_%d.bin", dump_dir, l);
            FILE *df = fopen(fname, "wb");
            if (df) {
                fwrite(x, sizeof(float), N * D_MODEL, df);
                fclose(df);
            }
        }
    }
    
    // Update KV cache length after processing all layers
    model->gqa_cache_len += T;

    // Save last hidden state for MTP speculative decode (if requested)
    // Captures BEFORE final RMSNorm — MTP head receives raw layer 39 output
    float *save_h = model->save_last_hidden;
    if (save_h && N > 0) {
        memcpy(save_h, x + (N - 1) * D_MODEL, D_MODEL * sizeof(float));
    }

    // Final RMSNorm
    if (model->norm_weight) {
        float *final_normed = (float *)malloc(N * D_MODEL * sizeof(float));
        wubu_rms_norm(B, T, D_MODEL, x, model->norm_weight, 1e-6f, final_normed);
        memcpy(x, final_normed, N * D_MODEL * sizeof(float));
        free(final_normed);
    }
    
    // Output projection
    // logits[t, v] = sum_k h[t,k] * output_weight[k, v]
    double t_out0 = wall_time();
    if (model->skip_output_proj) {
        // Copy final hidden states to logits buffer (caller does GPU output proj)
        for (int i = 0; i < N; i++) {
            memcpy(logits + i * model->vocab_size, x + i * D_MODEL,
                   D_MODEL * sizeof(float));
        }
    } else if (model->output_weight_q && model->output_weight_type != GGML_TYPE_F32) {
        // Q4_K quantized matmul path
        // For decode (N=1), quantized_matmul internal parallelizes across 248320 cols.
        // For prefill (N>1), use outer loop parallel with inner single-threaded matmul
        // to avoid 4x4=16 thread thrash on 4-core i5 (nested OMP oversubscription).
        #pragma omp parallel for if(N > 1)
        for (int i = 0; i < N; i++) {
            omp_set_num_threads(1);
            quantized_matmul(x + i * D_MODEL,
                             model->output_weight_q,
                             model->output_weight_type,
                             D_MODEL, model->vocab_size, 0,
                             logits + i * model->vocab_size);
        }
        omp_set_num_threads(omp_get_max_threads());
        // Compare against F32 SGEMM when output_weight is also loaded
        if (model->output_weight && getenv("VERBOSE_OUTPUT_PROJ")) {
            float *f32_logits = (float *)malloc(N * model->vocab_size * sizeof(float));
            #pragma omp parallel for collapse(2) if(N * model->vocab_size > 100000)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < model->vocab_size; j++) {
                    const float *h_i = x + i * D_MODEL;
                    float *log_i = f32_logits + i * model->vocab_size;
                    double sum = 0.0;
                    for (int k = 0; k < D_MODEL; k++)
                        sum += (double)h_i[k] * (double)model->output_weight[j * D_MODEL + k];
                    log_i[j] = (float)sum;
                }
            }
            double dot=0, n1=0, n2=0, max_e=0;
            for (int i = 0; i < N * model->vocab_size; i++) {
                dot += (double)logits[i] * (double)f32_logits[i];
                n1  += (double)logits[i] * (double)logits[i];
                n2  += (double)f32_logits[i] * (double)f32_logits[i];
                double e = fabs((double)logits[i] - (double)f32_logits[i]);
                if (e > max_e) max_e = e;
            }
            fprintf(stderr, "  [output proj] cos-sim Q4K vs F32 = %.10f, max_err=%.6f\n",
                    dot / (sqrt(n1) * sqrt(n2)), max_e);
            free(f32_logits);
        }
        double t_out1 = wall_time();
        if (getenv("PROFILE")) {
            fprintf(stderr, "  Output proj: %.3fms\n", (t_out1 - t_out0) * 1000.0);
        }
    } else {
        // Fallback: copy hidden states only (no output weight loaded)
        memcpy(logits, x, N * D_MODEL * sizeof(float));
    }
    
    free(x);
    free(normed);
    free(attn_out);
    free(normed2);
    free(ffn_out);
    free(prev_experts);
}

// ========== MTP Head ==========

bool wubu_mtp_load(mtp_head_t *mtp, const char *mtp_gguf_path,
                   gguf_ctx *main_ctx, const uint8_t *main_blob) {
    memset(mtp, 0, sizeof(*mtp));
    
    // Use the already-open main context (same model, same blob)
    // The MTP model is the same GGUF file as the main model
    gguf_ctx *ctx = main_ctx;
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    if (!ctx || !blob) {
        fprintf(stderr, "MTP: no context or blob available\n");
        return false;
    }
    
    // Verify this is an MTP model
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.40.nextn.hnorm.weight");
    if (!t) {
        fprintf(stderr, "MTP: no nextn tensors in model (not an MTP model?)\n");
        return false;
    }
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
    
    // eh_proj weight — dequant Q8_0 to F32 during init for fast SGEMM
    t = gguf_find_tensor(ctx, "blk.40.nextn.eh_proj.weight");
    if (!t) { fprintf(stderr, "MTP: missing eh_proj\n"); goto fail; }
    mtp->nextn_eh_proj_dim = (int64_t)t->dims[0];  // 4096
    int64_t eh_elems = (int64_t)t->dims[0] * (int64_t)t->dims[1];
    mtp->nextn_eh_proj_f32 = (float *)malloc(eh_elems * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, mtp->nextn_eh_proj_f32, eh_elems)) {
        fprintf(stderr, "MTP: failed to read eh_proj\n"); goto fail;
    }
    printf("MTP: eh_proj dequantized (%lld x %lld = %lld elems)\n",
           (long long)t->dims[0], (long long)t->dims[1], (long long)eh_elems);
    
    printf("MTP: nextn loaded (hnorm+enorm+eh_proj[%lldx%lld]+shared_head_norm)\n",
           (long long)t->dims[0], (long long)t->dims[1]);
    
    // Load blk.40 layer — use the MTP model's GGAUF for tensor offsets
    // We store pointers into the MTP model's data_blob for MoE and attn weights
    wubu_layer_t *blk40 = &mtp->blk40;
    memset(blk40, 0, sizeof(*blk40));
    blk40->layer_idx = 40;
    blk40->is_ssm = false;  // blk.40 is GQA (every 4th layer)
    
    // Load norms from MTP context (F32)
    // (blob already set above)
    
    t = gguf_find_tensor(ctx, "blk.40.attn_norm.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_norm\n"); goto fail; }
    blk40->attn_norm_weight = (float *)malloc(D_MODEL * sizeof(float));
    gguf_read_tensor_f32(ctx, t, blk40->attn_norm_weight, D_MODEL);
    
    t = gguf_find_tensor(ctx, "blk.40.post_attention_norm.weight");
    if (!t) { fprintf(stderr, "MTP: missing post_attn_norm\n"); goto fail; }
    blk40->post_attn_norm_weight = (float *)malloc(D_MODEL * sizeof(float));
    gguf_read_tensor_f32(ctx, t, blk40->post_attn_norm_weight, D_MODEL);
    
    // Load GQA weights (all Q5_K — type 13)
    // attn_q.weight [2048, 8192] — Q + gate fused
    t = gguf_find_tensor(ctx, "blk.40.attn_q.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_q\n"); goto fail; }
    blk40->gqa.attn_q_weight_q = blob + t->data_offset;
    blk40->gqa.attn_q_weight_type = t->ggml_type;
    
    t = gguf_find_tensor(ctx, "blk.40.attn_k.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_k\n"); goto fail; }
    blk40->gqa.attn_k_weight_q = blob + t->data_offset;
    blk40->gqa.attn_k_weight_type = t->ggml_type;
    
    t = gguf_find_tensor(ctx, "blk.40.attn_v.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_v\n"); goto fail; }
    blk40->gqa.attn_v_weight_q = blob + t->data_offset;
    blk40->gqa.attn_v_weight_type = t->ggml_type;
    
    t = gguf_find_tensor(ctx, "blk.40.attn_output.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_output\n"); goto fail; }
    blk40->gqa.attn_output_weight_q = blob + t->data_offset;
    blk40->gqa.attn_output_weight_type = t->ggml_type;
    
    // Q/K norms (F32)
    t = gguf_find_tensor(ctx, "blk.40.attn_q_norm.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_q_norm\n"); goto fail; }
    blk40->gqa.attn_q_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
    gguf_read_tensor_f32(ctx, t, blk40->gqa.attn_q_norm_weight, GQA_HEAD_DIM);
    
    t = gguf_find_tensor(ctx, "blk.40.attn_k_norm.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_k_norm\n"); goto fail; }
    blk40->gqa.attn_k_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
    gguf_read_tensor_f32(ctx, t, blk40->gqa.attn_k_norm_weight, GQA_HEAD_DIM);
    
    // Load MoE weights (quantized pointers into blob)
    moe_weights_t *moe = &blk40->moe;
    
    t = gguf_find_tensor(ctx, "blk.40.ffn_gate_inp.weight");
    if (t && blob) {
        // BF16 router — dequant to F32 during init
        int64_t n_router = (int64_t)t->dims[0] * t->dims[1];
        moe->ffn_gate_inp = (float *)malloc(n_router * sizeof(float));
        gguf_read_tensor_f32(ctx, t, moe->ffn_gate_inp, n_router);
    }
    
    t = gguf_find_tensor(ctx, "blk.40.ffn_gate_inp_shexp.weight");
    if (t && blob) {
        moe->ffn_gate_inp_shexp = (float *)malloc(D_MODEL * sizeof(float));
        gguf_read_tensor_f32(ctx, t, moe->ffn_gate_inp_shexp, D_MODEL);
    }
    
    // Routed experts: Q2_K (gate, up), Q3_K (down)
    t = gguf_find_tensor(ctx, "blk.40.ffn_gate_exps.weight");
    if (t && blob) { moe->ffn_gate_exps_q = blob + t->data_offset; moe->ffn_gate_exps_q_type = t->ggml_type; }
    t = gguf_find_tensor(ctx, "blk.40.ffn_up_exps.weight");
    if (t && blob) { moe->ffn_up_exps_q = blob + t->data_offset; moe->ffn_up_exps_q_type = t->ggml_type; }
    t = gguf_find_tensor(ctx, "blk.40.ffn_down_exps.weight");
    if (t && blob) { moe->ffn_down_exps_q = blob + t->data_offset; moe->ffn_down_exps_q_type = t->ggml_type; }
    
    // Shared expert: Q5_K (gate, up), Q6_K (down)
    t = gguf_find_tensor(ctx, "blk.40.ffn_gate_shexp.weight");
    if (t && blob) { moe->ffn_gate_shexp_q = blob + t->data_offset; moe->ffn_gate_shexp_q_type = t->ggml_type; }
    t = gguf_find_tensor(ctx, "blk.40.ffn_up_shexp.weight");
    if (t && blob) { moe->ffn_up_shexp_q = blob + t->data_offset; moe->ffn_up_shexp_q_type = t->ggml_type; }
    t = gguf_find_tensor(ctx, "blk.40.ffn_down_shexp.weight");
    if (t && blob) { moe->ffn_down_shexp_q = blob + t->data_offset; moe->ffn_down_shexp_q_type = t->ggml_type; }
    
    // Mark MoE as loaded
    if (moe->ffn_gate_exps_q && moe->ffn_up_exps_q && moe->ffn_down_exps_q) {
        moe->loaded = true;
        moe->load_from_blob = true;
    }
    
    printf("MTP: blk.40 loaded (GQA+MoE: Q5_K/Q2_K/Q3_K/Q6_K)\n");
    
    // Allocate KV cache for blk.40
    mtp->k_cache = (float *)calloc(GQA_MAX_CTX * GQA_KV_DIM, sizeof(float));
    mtp->v_cache = (float *)calloc(GQA_MAX_CTX * GQA_KV_DIM, sizeof(float));
    mtp->cache_len = 0;
    
    mtp->loaded = true;
    return true;
    
fail:
    wubu_mtp_free(mtp);
    return false;
}

int wubu_mtp_draft_forward(wubu_model_t *model,
                           const float *x,
                           const float *token_embd, int B,
                           float *logits_out) {
    if (!model->mtp.loaded) return 0;
    
    mtp_head_t *mtp = &model->mtp;
    wubu_layer_t *blk40 = &mtp->blk40;
    const int vs = model->vocab_size;
    
    // Per-draft buffers (reuse across B to avoid mallocs)
    float *h_norm = (float *)malloc(D_MODEL * sizeof(float));
    float *e_norm = (float *)malloc(D_MODEL * sizeof(float));
    float *concat = (float *)malloc(2 * D_MODEL * sizeof(float));
    float *cur = (float *)malloc(D_MODEL * sizeof(float));
    float *temp_attn = (float *)malloc(D_MODEL * sizeof(float));
    float *temp_ffn = (float *)malloc(D_MODEL * sizeof(float));
    float *temp_norm = (float *)malloc(D_MODEL * sizeof(float));
    
    if (!h_norm || !e_norm || !concat || !cur || !temp_attn || !temp_ffn || !temp_norm) {
        fprintf(stderr, "MTP draft: alloc failed\n");
        free(h_norm); free(e_norm); free(concat); free(cur);
        free(temp_attn); free(temp_ffn); free(temp_norm);
        return 0;
    }
    
    // Step 1: h_norm = rms_norm(x, hnorm)
    wubu_rms_norm(1, 1, D_MODEL, x, mtp->nextn_hnorm, 1e-6f, h_norm);
    
    // Process each draft token
    for (int b = 0; b < B; b++) {
        const float *embd_b = token_embd + b * D_MODEL;
        float *logits_b = logits_out + b * vs;
        
        // Step 2: e_norm = rms_norm(token_embd[b], enorm)
        wubu_rms_norm(1, 1, D_MODEL, embd_b, mtp->nextn_enorm, 1e-6f, e_norm);
        
        // Step 3: concat = [e_norm | h_norm] (llama.cpp order: ggml_concat(e_norm, h_norm, 0))
        memcpy(concat, e_norm, D_MODEL * sizeof(float));
        memcpy(concat + D_MODEL, h_norm, D_MODEL * sizeof(float));
        
        // Step 4: cur = eh_proj @ concat (F32 SGEMM)
        for (int j = 0; j < D_MODEL; j++) {
            double sum = 0.0;
            for (int k = 0; k < mtp->nextn_eh_proj_dim; k++)
                sum += (double)concat[k] * (double)mtp->nextn_eh_proj_f32[j * mtp->nextn_eh_proj_dim + k];
            cur[j] = (float)sum;
        }
        
        // Step 5: Forward through blk.40 (GQA+MoE)
        // Pre-attention RMSNorm
        wubu_rms_norm(1, 1, D_MODEL, cur, blk40->attn_norm_weight, 1e-6f, temp_norm);
        
        // GQA forward with KV cache
        float *k_out = mtp->k_cache + (mtp->cache_len + b) * GQA_KV_DIM;
        float *v_out = mtp->v_cache + (mtp->cache_len + b) * GQA_KV_DIM;
        wubu_gqa_forward(temp_norm, 1, 1, &blk40->gqa, temp_attn,
                         mtp->k_cache, mtp->v_cache, mtp->cache_len + b,
                         k_out, v_out);
        
        // Residual
        for (int i = 0; i < D_MODEL; i++) cur[i] += temp_attn[i];
        
        // Post-attention RMSNorm
        wubu_rms_norm(1, 1, D_MODEL, cur, blk40->post_attn_norm_weight, 1e-6f, temp_norm);
        
        // MoE forward
        if (blk40->moe.loaded) {
            wubu_moe_forward(temp_norm, 1, 1, &blk40->moe, temp_ffn, NULL);
        } else {
            memcpy(temp_ffn, temp_norm, D_MODEL * sizeof(float));
        }
        
        // Residual
        for (int i = 0; i < D_MODEL; i++) cur[i] += temp_ffn[i];
        
        // Step 6: shared_head_norm
        wubu_rms_norm(1, 1, D_MODEL, cur, mtp->nextn_shared_head_norm, 1e-6f, temp_norm);
        
        // Step 7: output projection (via main model's output.weight)
        if (model->output_weight_q) {
            quantized_matmul(temp_norm, model->output_weight_q, model->output_weight_type,
                            D_MODEL, vs, 0, logits_b);
        } else {
            memset(logits_b, 0, vs * sizeof(float));
        }
    }
    
    // Update cache length
    mtp->cache_len += B;
    
    free(h_norm); free(e_norm); free(concat); free(cur);
    free(temp_attn); free(temp_ffn); free(temp_norm);
    
    return B;
}

void wubu_mtp_free(mtp_head_t *mtp) {
    if (!mtp || !mtp->loaded) return;
    free(mtp->nextn_hnorm);
    free(mtp->nextn_enorm);
    free(mtp->nextn_shared_head_norm);
    free(mtp->nextn_eh_proj_f32);
    // blk.40 GQA norms
    free(mtp->blk40.attn_norm_weight);
    free(mtp->blk40.post_attn_norm_weight);
    free(mtp->blk40.gqa.attn_q_norm_weight);
    free(mtp->blk40.gqa.attn_k_norm_weight);
    // blk.40 MoE (blob-backed so only F32 pointers freed)
    if (!mtp->blk40.moe.load_from_blob) {
        free(mtp->blk40.moe.ffn_gate_inp);
        free(mtp->blk40.moe.ffn_gate_inp_shexp);
    } else {
        free(mtp->blk40.moe.ffn_gate_inp);
        free(mtp->blk40.moe.ffn_gate_inp_shexp);
    }
    free(mtp->k_cache);
    free(mtp->v_cache);
    memset(mtp, 0, sizeof(*mtp));
}

// ========== State Save/Restore for Speculative Decode ==========

bool wubu_model_checkpoint(wubu_model_t *model) {
    // Lazy allocation on first call
    if (!model->ssm_states_saved) {
        int n_layers = model->n_layers;
        int ssm_sz = n_layers * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
        int conv_sz = n_layers * (CONV_KERNEL - 1) * CONV_DIM;
        model->ssm_states_saved = (float *)malloc((ssm_sz + conv_sz) * sizeof(float));
        if (!model->ssm_states_saved) return false;
        model->conv_states_saved = model->ssm_states_saved + ssm_sz;
    }
    // Save SSM states + conv states
    int n_layers = model->n_layers;
    int ssm_sz = n_layers * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    int conv_sz = n_layers * (CONV_KERNEL - 1) * CONV_DIM;
    memcpy(model->ssm_states_saved, model->ssm_states, (ssm_sz + conv_sz) * sizeof(float));
    // Save cache lengths
    model->gqa_cache_len_saved = model->gqa_cache_len;
    model->mtp_cache_len_saved = model->mtp.cache_len;
    return true;
}

void wubu_model_rollback(wubu_model_t *model) {
    if (!model->ssm_states_saved) return;
    int n_layers = model->n_layers;
    int ssm_sz = n_layers * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    int conv_sz = n_layers * (CONV_KERNEL - 1) * CONV_DIM;
    // Restore SSM states + conv states
    memcpy(model->ssm_states, model->ssm_states_saved, (ssm_sz + conv_sz) * sizeof(float));
    // Restore cache lengths
    model->gqa_cache_len = model->gqa_cache_len_saved;
    model->mtp.cache_len = model->mtp_cache_len_saved;
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
                size_t nr = fread(embd + i * D_MODEL, sizeof(float), D_MODEL, f);
                (void)nr;
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
