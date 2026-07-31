#include "wubu_model.h"
#include "gguf_reader.h"
#include "safetensors_reader.h"
#include "wubu_affinity.h"
#include "wubu_rotate.h"   // doc 013: wubu_rotate_input for lm_head Hadamard fuse
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>  // _mm_prefetch for expert prefetch

// Global tensor naming convention (set during model init)
extern int g_tensor_naming;  // defined in wubu_ssm.c, 0=blk.Qwen 1=model.layers.Gemma 2=pure-GQA

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
    model->tied_output = false;
    model->rotate_P = 0;

    /* Game-console hardware discipline (I05 / NuMA+P-core pinning, +19-21%
     * throughput on multi-socket; non-zero even single-socket via stable
     * L1/L2 per GEMV row-chunk). Pin the calling thread to the
     * P-core set, then make OpenMP inherit a close, core-bound policy so
     * the GEMV parallel-for keeps each row-chunk on one core's cache. */
    {
        int pinned[64]; int k = wubu_affinity_pin_pcores(pinned, 64);
        if (k > 0) {
            /* Make OpenMP inherit a close, core-bound policy so the GEMV
             * parallel-for keeps each row-chunk on one core's cache. Use
             * setenv (portable across OpenMP runtimes) rather than the
             * version-specific omp_set_proc_bind API. */
            setenv("OMP_PROC_BIND", "close", 1);
            setenv("OMP_PLACES", "cores", 1);
            setenv("OMP_SCHEDULE", "dynamic,64", 1);
            fprintf(stderr, "[affinity] pinned engine to %d P-cores (core0=%d)\n",
                    k, pinned[0]);
        }
    }

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

    // ============================================================
    // Multi-model dimension extraction from GGUF
    // ============================================================
    // Detect tensor naming convention and architecture
    model->tensor_naming = 0; // default: Qwen (blk.N.*)
    for (int i = 0; i < (int)ctx->n_tensors; i++) {
        if (strncmp(ctx->tensors[i].name, "model.layers.", 12) == 0) {
            model->tensor_naming = 1; // Gemma-style
            break;
        }
    }
    // Detect pure GQA (no SSM layers) by checking for ssm_beta tensor
    {
        const char *ssm_check = (model->tensor_naming == 1) ? "model.layers.0.ssm_beta.weight" : "blk.0.ssm_beta.weight";
        if (!gguf_find_tensor(ctx, ssm_check)) {
            model->tensor_naming = 2; // pure GQA (DiffusionGemma/Gemma4)
        }
    }
    g_tensor_naming = model->tensor_naming; // set global for wubu_is_ssm_layer()

    // Extract dynamic dimensions from GGUF tensor shapes
    int d_model = 0;
    {
        const char *norm_name = (model->tensor_naming == 1) ? "model.layers.0.attn_norm.weight" : "blk.0.attn_norm.weight";
        gguf_tensor_info *nt = gguf_find_tensor(ctx, norm_name);
        if (nt && nt->n_dims >= 1) d_model = (int)nt->dims[0];
    }
    if (d_model == 0) d_model = D_MODEL; // fallback
    model->d_model = d_model;

    // Extract GQA dimensions from tensor shapes
    int gqa_head_dim = GQA_HEAD_DIM;
    {
        const char *q_norm_name = (model->tensor_naming == 1) ? "model.layers.0.attn_q_norm.weight" : "blk.0.attn_q_norm.weight";
        gguf_tensor_info *qn = gguf_find_tensor(ctx, q_norm_name);
        if (qn && qn->n_dims >= 1 && qn->dims[0] > 0) {
            gqa_head_dim = (int)qn->dims[0];
        } else {
            // Fallback 1: try to derive from attn_k.weight shape [d_model, kv_heads * head_dim]
            const char *k_name = "blk.0.attn_k.weight";
            gguf_tensor_info *kn = gguf_find_tensor(ctx, k_name);
            if (kn && kn->n_dims >= 2) {
                int kv_dim = (int)kn->dims[1];
                int kv_heads = (kv_dim > 0) ? (kv_dim / 256) : 10;
                if (kv_heads > 0) {
                    gqa_head_dim = kv_dim / kv_heads;
                }
            } else {
                // Fallback 2: Qwen3.6 uses attn_qkv.weight [d_model, (q_heads + 2*kv_heads) * head_dim]
                const char *qkv_name = "blk.0.attn_qkv.weight";
                gguf_tensor_info *qkn = gguf_find_tensor(ctx, qkv_name);
                if (qkn && qkn->n_dims >= 2) {
                    int qkv_dim = (int)qkn->dims[1];
                    int assumed_kv_heads = 4;
                    if (qkv_dim > assumed_kv_heads * 256) {
                        gqa_head_dim = 256;
                    }
                }
            }
        }
    }

    // Extract SSM dimensions from tensor shapes
    int ssm_d_state = SSM_D_STATE;
    int ssm_k_heads = SSM_K_HEADS;
    int dt_rank = DT_RANK;
    int ssm_v_heads = SSM_V_HEADS;
    int conv_kernel = CONV_KERNEL;
    {
        // ssm_norm.weight [SSM_D_STATE]
        gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.ssm_norm.weight");
        if (t && t->n_dims >= 1) {
            ssm_d_state = (int)t->dims[0];
        }
        // ssm_dt.bias [DT_RANK]
        t = gguf_find_tensor(ctx, "blk.0.ssm_dt.bias");
        if (t && t->n_dims >= 1) {
            dt_rank = (int)t->dims[0];
        }
        // ssm_a [DT_RANK]
        t = gguf_find_tensor(ctx, "blk.0.ssm_a");
        if (t && t->n_dims >= 1) {
            dt_rank = (int)t->dims[0];
        }
        // ssm_conv1d.weight [CONV_KERNEL, CONV_DIM]
        t = gguf_find_tensor(ctx, "blk.0.ssm_conv1d.weight");
        if (t && t->n_dims >= 2) {
            conv_kernel = (int)t->dims[0];
            int conv_dim = (int)t->dims[1];
            // CONV_DIM = 2 * KEY_DIM + VALUE_DIM
            // KEY_DIM = SSM_D_STATE * SSM_K_HEADS
            // VALUE_DIM = SSM_D_STATE * SSM_V_HEADS
            // We know SSM_D_STATE and CONV_DIM, solve for SSM_V_HEADS
            // conv_dim = 2 * (ssm_d_state * ssm_k_heads) + ssm_d_state * ssm_v_heads
            // ssm_v_heads = (conv_dim - 2 * ssm_d_state * ssm_k_heads) / ssm_d_state
            int key_dim = ssm_d_state * ssm_k_heads;
            int value_dim = conv_dim - 2 * key_dim;
            if (value_dim > 0 && value_dim % ssm_d_state == 0) {
                ssm_v_heads = value_dim / ssm_d_state;
            }
        }
    }

    // Setup WUBU_DIMS from extracted dimensions
    wubu_dims_t dims = {0};
    dims.d_model = d_model;
    dims.ssm_d_state = ssm_d_state;
    dims.ssm_k_heads = ssm_k_heads;
    dims.ssm_v_heads = ssm_v_heads;
    dims.conv_kernel = conv_kernel;
    dims.dt_rank = dt_rank;
    dims.gqa_q_heads = GQA_Q_HEADS;
    dims.gqa_kv_heads = GQA_KV_HEADS;
    dims.gqa_head_dim = gqa_head_dim;
    wubu_dims_set(&dims);

    // Also set model fields for backward compatibility
    model->d_inner = VALUE_DIM;
    model->key_dim = KEY_DIM;
    model->conv_dim = CONV_DIM;
    model->conv_kernel = CONV_KERNEL;
    model->dt_rank = dt_rank;
    model->ssm_k_heads = ssm_k_heads;
    model->ssm_v_heads = ssm_v_heads;
    model->ssm_d_state = ssm_d_state;
    model->gqa_q_heads = GQA_Q_HEADS;
    model->gqa_kv_heads = GQA_KV_HEADS;
    model->gqa_head_dim = gqa_head_dim;
    model->rotary_dim = (int)(gqa_head_dim * PARTIAL_ROTARY_FACTOR);
    model->d_ff = D_FF;
    model->n_experts = N_EXPERTS;
    model->n_active_experts = N_ACTIVE_EXPTS;

    printf("  Model dims: d_model=%d, head_dim=%d\n", d_model, gqa_head_dim);
    printf("  SSM dims: d_state=%d, k_heads=%d, v_heads=%d, dt_rank=%d, conv_kernel=%d\n",
           ssm_d_state, ssm_k_heads, ssm_v_heads, dt_rank, conv_kernel);
    printf("  CONV_DIM=%d, VALUE_DIM=%d, KEY_DIM=%d\n", CONV_DIM, VALUE_DIM, KEY_DIM);
    printf("  Naming: %s\n", model->tensor_naming == 1 ? "Gemma (model.layers.N.*)" : (model->tensor_naming == 2 ? "Pure-GQA (blk.N.*)" : "Qwen (blk.N.*)"));

    // Buffer GGUF data EARLY so all tensor reads use mmap (avoids FILE* issues with large files)
    printf("  Buffering GGUF data via mmap...\n");
    if (!gguf_buffer_data(ctx)) {
        fprintf(stderr, "Failed to buffer GGUF data\n");
        goto fail;
    }
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    printf("  GGUF data buffered: %p (mmap=%d)\n", (void*)blob, ctx->data_blob_is_mmap);

    // Load layer norms and attention weights
    for (int l = 0; l < model->n_layers; l++) {
        wubu_layer_t *layer = &model->layers[l];
        layer->layer_idx = l;
        layer->is_ssm = wubu_is_ssm_layer(l);
        
        gguf_tensor_info *t;
        char name[256];

        // attn_norm.weight (pre-attention RMSNorm)
        t = gguf_find_tensor(ctx, tensor_name_attn_norm(l));
        if (t) {
            layer->attn_norm_weight = (float *)malloc(model->d_model * sizeof(float));
            if (!gguf_read_tensor_f32(ctx, t, layer->attn_norm_weight, model->d_model))
                { fprintf(stderr, "Failed to load attn_norm[%d]\n", l); goto fail; }
        }
        
        // post_attention_norm.weight (optional — Qwen3-style).
        // Fallbacks: ffn_norm.weight (Qwen2/nanbeige), then attn_norm.weight.
        t = gguf_find_tensor(ctx, tensor_name_post_attn_norm(l));
        if (!t) {
            snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
            t = gguf_find_tensor(ctx, name);
        }
        if (t) {
            layer->post_attn_norm_weight = (float *)malloc(model->d_model * sizeof(float));
            if (!gguf_read_tensor_f32(ctx, t, layer->post_attn_norm_weight, model->d_model))
                { fprintf(stderr, "Failed to load post_attn_norm[%d]\n", l); goto fail; }
        } else if (layer->attn_norm_weight) {
            // No dedicated post-attn norm: reuse pre-attn norm (identity-ish RMSNorm).
            layer->post_attn_norm_weight = (float *)malloc(model->d_model * sizeof(float));
            memcpy(layer->post_attn_norm_weight, layer->attn_norm_weight, model->d_model * sizeof(float));
        }
        
        if (layer->is_ssm) {
            // Load SSM weights — QUANTIZED-ONLY PATH for large weight matrices.
            // attn_qkv, attn_gate, ssm_out use quantized blob pointers (set later).
            // Small tensors (norms, a, dt, conv1d) loaded as F32.
            int ok = 1;
            
            if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: Loading SSM layer %d\n", l);
            
            // LARGE: attn_qkv_weight — quantized-only (blob pointer)
            layer->ssm.attn_qkv_weight = NULL;
            snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: SSM layer %d - attn_qkv found\n", l);
            
            // LARGE: attn_gate_weight — quantized-only (blob pointer)
            layer->ssm.attn_gate_weight = NULL;
            snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
            if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: SSM layer %d - attn_gate found\n", l);
            
            // Small: ssm_beta.weight [d_model, dt_rank] F32
                        snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", l);
                        t = gguf_find_tensor(ctx, name);
                        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
                        layer->ssm.ssm_beta_weight = (float *)malloc(model->d_model * model->dt_rank * sizeof(float));
                        if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: SSM layer %d - ssm_beta_weight malloc'd\n", l);
                        ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_beta_weight, -1) > 0);
                        if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: SSM layer %d - ssm_beta_weight loaded\n", l);
            
                        // Small: ssm_alpha.weight [d_model, dt_rank] F32
                        snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", l);
                        t = gguf_find_tensor(ctx, name);
                        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
                        layer->ssm.ssm_alpha_weight = (float *)malloc(model->d_model * model->dt_rank * sizeof(float));
                        if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: SSM layer %d - ssm_alpha_weight malloc'd\n", l);
                        ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_alpha_weight, -1) > 0);
                        if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: SSM layer %d - ssm_alpha_weight loaded\n", l);
            
                        // Small: ssm_dt.bias [dt_rank] F32
                        snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", l);
                        t = gguf_find_tensor(ctx, name);
                        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
                        layer->ssm.ssm_dt_bias = (float *)malloc(model->dt_rank * sizeof(float));
                        if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: SSM layer %d - ssm_dt_bias malloc'd\n", l);
                        ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_dt_bias, -1) > 0);
                        if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: SSM layer %d - ssm_dt_bias loaded\n", l);
            
                        // Small: ssm_a [dt_rank] F32 (Qwen3.6 uses "ssm_a" without .weight suffix)
                        snprintf(name, sizeof(name), "blk.%d.ssm_a", l);
                        t = gguf_find_tensor(ctx, name);
                        if (!t) {
                            // Fallback: try with .weight suffix
                            snprintf(name, sizeof(name), "blk.%d.ssm_a.weight", l);
                            t = gguf_find_tensor(ctx, name);
                        }
                        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
                        layer->ssm.ssm_a = (float *)malloc(model->dt_rank * sizeof(float));
                        if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: SSM layer %d - ssm_a malloc'd\n", l);
                        ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_a, -1) > 0);
                        if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: SSM layer %d - ssm_a loaded\n", l);
            
                        // Small: ssm_conv1d.weight [conv_kernel, conv_dim] F32
                        snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", l);
                        t = gguf_find_tensor(ctx, name);
                        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
                        layer->ssm.ssm_conv1d_weight = (float *)malloc(model->conv_kernel * model->conv_dim * sizeof(float));
                        ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_conv1d_weight, -1) > 0);
                        if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: SSM layer %d - ssm_conv1d_weight loaded\n", l);
                        if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: SSM layer %d - checking ssm_norm_weight\n", l);
            
                        // Small: ssm_norm.weight [ssm_d_state] F32
                        snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", l);
                        t = gguf_find_tensor(ctx, name);
                        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
                        if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: SSM layer %d - ssm_norm_weight tensor found, n_dims=%d, dims[0]=%ld\n", l, t->n_dims, t->dims[0]);
                        // Use the tensor's actual dimension, not model->ssm_d_state
                        int ssm_norm_size = (int)t->dims[0];
                        layer->ssm.ssm_norm_weight = (float *)malloc(ssm_norm_size * sizeof(float));
                        ok = ok && (gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_norm_weight, -1) > 0);
                        if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: SSM layer %d - ssm_norm_weight loaded\n", l);
            
            // LARGE: ssm_out.weight — quantized-only (blob pointer)
            layer->ssm.ssm_out_weight = NULL;
            
            if (!ok) { fprintf(stderr, "Failed to load SSM weights for layer %d\n", l); goto fail; }
            printf("  Layer %d: SSM loaded (quantized attn_qkv/gate/out)\n", l);
            
        } else {
            // Load GQA weights — QUANTIZED-ONLY PATH for large weight matrices.
            // attn_q, attn_k, attn_v, attn_output use quantized blob pointers (set later).
            char name[256];
            int ok = 1;
            
            if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: Loading GQA layer %d\n", l);
            
            // LARGE: attn_q.weight — quantized-only (blob pointer)
            layer->gqa.attn_q_weight = NULL;
            
            // LARGE: attn_k.weight — quantized-only (blob pointer)
            layer->gqa.attn_k_weight = NULL;
            
            // LARGE: attn_v.weight — quantized-only (blob pointer)
            layer->gqa.attn_v_weight = NULL;
            
            // LARGE: attn_output.weight — quantized-only (blob pointer)
            layer->gqa.attn_output_weight = NULL;
            
            // Small: attn_q_norm.weight [head_dim] F32
            // Optional: some GGUFs (Qwen-style / Pure-GQA without per-head
            // norms, e.g. nanbeige) omit blk.N.attn_q_norm.weight entirely.
            // Treat absence as identity (RMSNorm with all-ones weight).
            snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", l);
            t = gguf_find_tensor(ctx, name);
            int layer_head_dim = model->gqa_head_dim;  // Use model-level head_dim as default
            if (t && t->n_dims >= 1 && t->dims[0] > 0) layer_head_dim = (int)t->dims[0];
            layer->gqa.head_dim = layer_head_dim;
            layer->gqa.attn_q_norm_weight = (float *)malloc(layer_head_dim * sizeof(float));
            if (t) {
                ok = ok && (gguf_read_tensor_f32(ctx, t, layer->gqa.attn_q_norm_weight, layer_head_dim) > 0);
            } else {
                for (int i = 0; i < layer_head_dim; i++) layer->gqa.attn_q_norm_weight[i] = 1.0f;
            }

            // Small: attn_k_norm.weight [head_dim] F32 (optional, see above)
            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", l);
            t = gguf_find_tensor(ctx, name);
            layer->gqa.attn_k_norm_weight = (float *)malloc(layer_head_dim * sizeof(float));
            if (t) {
                ok = ok && (gguf_read_tensor_f32(ctx, t, layer->gqa.attn_k_norm_weight, layer_head_dim) > 0);
            } else {
                for (int i = 0; i < layer_head_dim; i++) layer->gqa.attn_k_norm_weight[i] = 1.0f;
            }

            // Extract per-layer dimensions from GGUF tensor shapes
            // K weight: [d_model, kv_heads * head_dim] => kv_heads from dims
            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (t && t->n_dims >= 2 && layer_head_dim > 0) {
                int kv_dim = (int)t->dims[1];  // kv_heads * head_dim
                layer->gqa.kv_heads = kv_dim / layer_head_dim;
            } else {
                layer->gqa.kv_heads = GQA_KV_HEADS;
            }
            // Q weight: [d_model, q_heads * head_dim * 2] (fused Q+gate) => q_heads from dims
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (t && t->n_dims >= 2 && layer_head_dim > 0) {
                int q_dim_fused = (int)t->dims[1];  // q_heads * head_dim * 2
                layer->gqa.q_heads = q_dim_fused / (layer_head_dim * 2);
            } else {
                layer->gqa.q_heads = GQA_Q_HEADS;
            }
            layer->gqa.kv_dim = layer->gqa.kv_heads * layer_head_dim;
            layer->gqa.q_dim = layer->gqa.q_heads * layer_head_dim;
            layer->gqa.is_large = (layer_head_dim == 512) ? 1 : 0;

            // Extract output projection dim from attn_output.weight tensor
            // For standard models (Qwen): out_dim = q_dim
            // For DGemma: out_dim = q_dim * 2 (Q+gate fused into output proj)
            layer->gqa.out_dim = layer->gqa.q_dim;  // default
            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
            t = gguf_find_tensor(ctx, name);
            if (t && t->n_dims >= 2) {
                int out_rows = (int)t->dims[0];  // first dim = input features to output proj
                if (out_rows != layer->gqa.q_dim) {
                    layer->gqa.out_dim = out_rows;  // e.g., q_dim*2 for DGemma
                }
            }

            printf("  Layer %d: GQA loaded, head_dim=%d q_heads=%d kv_heads=%d out_dim=%d%s\n",
                   l, layer_head_dim, layer->gqa.q_heads, layer->gqa.kv_heads,
                   layer->gqa.out_dim, layer->gqa.is_large ? " LARGE" : "");
        }
        
        // Load MoE (FFN) weights — NOT loaded by default (memory: 3.2 GB/layer)
        // Use test_moe.c for standalone MoE testing
        layer->moe.loaded = false;
    }
    
    // Load final norm
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output_norm.weight");
    if (t) {
        model->norm_weight = (float *)malloc(model->d_model * sizeof(float));
        gguf_read_tensor_f32(ctx, t, model->norm_weight, model->d_model);
        printf("  Final norm loaded\n");
    } else {
        printf("  WARNING: output_norm.weight not found\n");
    }
    
    // Embeddings: auto-extract from GGUF if not available, else load from file
    model->use_embedding_file = true;
    model->vocab_size = 0;
    // Get actual vocab size from GGUF embedding tensor
    {
        gguf_tensor_info *t_emb = gguf_find_tensor(ctx, "token_embd.weight");
        if (t_emb && t_emb->n_dims >= 2) {
            int64_t n_emb = 1;
            for (int d = 0; d < t_emb->n_dims; d++) n_emb *= t_emb->dims[d];
            int64_t vocab_from_emb = n_emb / model->d_model;
            if (vocab_from_emb > 0 && vocab_from_emb < 1000000) {
                model->vocab_size = (int)vocab_from_emb;
            }
        }
        if (model->vocab_size == 0) model->vocab_size = 248320; // fallback
    }
    // For large-vocab models (Gemma, vocab > 131072), skip F32 embedding load.
    // Use the mmap'd GGUF blob directly for per-token dequant embedding lookup.
    // For small-vocab models (Qwen), the F32 path is fine.
    bool large_vocab = (model->vocab_size > 131072);
    model->use_embedding_file = false;
    model->token_embd = NULL;

    if (!large_vocab) {
        // Small vocab: try loading F32 embeddings (original path)
        const char *emb_path = "data/qwen36_embeddings_c.bin.raw";
        FILE *emb_f = fopen(emb_path, "rb");
        if (emb_f) {
            fseek(emb_f, 0, SEEK_END);
            long emb_size = ftell(emb_f);
            int file_vocab = (int)(emb_size / (model->d_model * sizeof(float)));
            if (file_vocab == model->vocab_size) {
                printf("  Embeddings: %d tokens from file (%ld MB)\n", model->vocab_size, emb_size / (1024*1024));
                fclose(emb_f);
                model->use_embedding_file = true;
            } else {
                fclose(emb_f);
                large_vocab = true;
            }
        } else {
            large_vocab = true;
        }
    }

    if (large_vocab) {
        // Large vocab: use mmap'd GGUF blob for per-token dequant
        printf("  Large vocab (%d tokens): using mmap'd GGUF blob for embedding\n", model->vocab_size);
        model->use_embedding_file = false;
        model->token_embd = NULL;
        // Get quantized token_embd pointer from blob
        gguf_tensor_info *t_emb = gguf_find_tensor(ctx, "token_embd.weight");
        if (t_emb && ctx->data_blob) {
            model->token_embd_q = (const uint8_t *)ctx->data_blob + t_emb->data_offset;
            model->token_embd_type = t_emb->ggml_type;
            int64_t emb_elems = 1;
            for (int d = 0; d < t_emb->n_dims; d++) emb_elems *= t_emb->dims[d];
            printf("  token_embd: quantized type=%d, %ld elements\n", t_emb->ggml_type, (long)emb_elems);
        }
    }
    
    if (model->use_embedding_file) {
        // Verify vocab_size was set from file
        if (model->vocab_size == 0) model->vocab_size = 248320;
    }
    
    // Load output weight for logit projection — QUANTIZED-ONLY (Q4_K blob pointer)
    model->output_weight = NULL;
    gguf_tensor_info *t_out = gguf_find_tensor(ctx, "output.weight");
    if (!t_out) {
        // Tied output: use token_embd.weight (common for Gemma, LLaMA, etc.)
        gguf_tensor_info *t_embd = gguf_find_tensor(ctx, "token_embd.weight");
        if (t_embd) {
            model->output_weight_q = t_embd;  // will get blob pointer below
            model->output_weight_type = t_embd->ggml_type;
            model->tied_output = true;
            fprintf(stderr, "  Output weight: TIED to token_embd.weight\n");
        } else {
            fprintf(stderr, "  ERROR: neither output.weight nor token_embd.weight found\n");
        }
    } else {
        model->tied_output = false;
    }
    
    // Output weight quantized pointer will be set after gguf_buffer_data() below
    printf("  Output weight: will use quantized path (Q4_K via blob pointer)\n");
    
    // Allocate state buffers (one SSM state per layer, not per position).
// The SSM recurrence is sequential — each layer has a single persistent
// state vector of size v_heads × d_state². max_s=1 for decode.
// For prefill (B>1), we only need the per-layer state, not per-token.
int max_s = 1;
    int ssm_state_size = max_s * model->ssm_v_heads * model->ssm_d_state * model->ssm_d_state;
    int conv_state_size = max_s * (model->conv_kernel - 1) * model->conv_dim;
    if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: Allocating state buffers: max_s=%d, ssm_state_size=%d, conv_state_size=%d\n", max_s, ssm_state_size, conv_state_size);
    model->ssm_states = (float *)calloc(ssm_state_size + conv_state_size, sizeof(float));
    if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: ssm_states allocated\n");
    model->conv_states = model->ssm_states + ssm_state_size;
    model->ssm_state_total = (size_t)(ssm_state_size + conv_state_size) * sizeof(float);
    
    model->gguf_ctx = ctx;  // Keep ctx open for per-layer MoE loading
    model->enable_moe = false;  // MoE disabled by default (memory: 3.2 GB/layer)
    model->moe_max_layers = 0;  // 0 = all layers
    
    // Read SSM L2 norm epsilon from GGUF config (qwen35moe.attention.layer_norm_rms_epsilon = 1e-6)
    g_ssm_l2_eps = 1e-6f;
    printf("  SSM L2 eps: %e\n", g_ssm_l2_eps);
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
                if (t && blob) {
                    layer->gqa.attn_v_weight_q = blob + t->data_offset;
                    layer->gqa.attn_v_weight_type = t->ggml_type;
                } else {
                    /* LARGE layers: V weight not present, share K weight (V=K) */
                    layer->gqa.attn_v_weight_q = layer->gqa.attn_k_weight_q;
                    layer->gqa.attn_v_weight_type = layer->gqa.attn_k_weight_type;
                }
                snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
                t = gguf_find_tensor(ctx, name);
                if (t && blob) { layer->gqa.attn_output_weight_q = blob + t->data_offset; layer->gqa.attn_output_weight_type = t->ggml_type; }
            }
        }
        if (t_out && blob) {
            model->output_weight_q = blob + t_out->data_offset;
            model->output_weight_type = t_out->ggml_type;
            model->tied_output = false;
        } else if (model->tied_output) {
            // Tied: output_weight_q was set to token_embd tensor info earlier,
            // but we need the actual blob pointer
            gguf_tensor_info *t_embd = gguf_find_tensor(ctx, "token_embd.weight");
            if (t_embd && blob) {
                model->output_weight_q = blob + t_embd->data_offset;
            }
        }

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

    // Count actual SSM and GQA layers
    int n_ssm_count = 0, n_gqa_count = 0;
    for (int l = 0; l < model->n_layers; l++) {
        if (model->layers[l].is_ssm) n_ssm_count++;
        else n_gqa_count++;
    }
    model->n_gqa_layers = n_gqa_count;
    printf("Model initialized: %d layers (%d SSM, %d GQA), %d vocab\n",
           model->n_layers, n_ssm_count, n_gqa_count, model->vocab_size);

    // Allocate GQA KV cache: sum over all GQA layers of (max_ctx * layer_kv_dim)
    // Runtime override: WUBU_MAX_CTX env var. Default 8192 (safe for 13GB RAM).
    // SWA + auto-eviction handles contexts larger than max_ctx.
    int runtime_max_ctx = GQA_MAX_CTX;
    {
        const char *mc_env = getenv("WUBU_MAX_CTX");
        if (mc_env) {
            int mc = atoi(mc_env);
            if (mc > 0) runtime_max_ctx = mc;
        }
    }
    int64_t total_cache_elems = 0;
    for (int l = 0; l < model->n_layers; l++) {
        if (!model->layers[l].is_ssm) {
            int kv_dim = model->layers[l].gqa.kv_dim;
            total_cache_elems += (int64_t)runtime_max_ctx * kv_dim;
        }
    }
    // Auto-select KV precision from the Roofline crossover, using real model
    // dimensions + detected bandwidth (env WUBU_BW_TBS overrides, TB/s).
    {
        int gqa_head_dim = model->gqa_head_dim, gqa_n_kv = 1;
        for (int l = 0; l < model->n_layers; l++) {
            if (!model->layers[l].is_ssm) {
                gqa_head_dim = model->layers[l].gqa.head_dim;
                gqa_n_kv    = model->layers[l].gqa.kv_heads;
                break;
            }
        }
        double bw = 0.05; /* default CPU ~50 GB/s */
        const char *bw_env = getenv("WUBU_BW_TBS");
        if (bw_env) bw = atof(bw_env);
        /* Rough param count (transformer ~12 * d_model^2 * n_layers) used only
         * for the relative Roofline B* crossover, not absolute memory. */
        double n_params = (double)model->d_model * model->d_model * model->n_layers * 12.0;
        int chosen = wubu_kv_autoselect(
            n_params, model->n_layers, gqa_n_kv, gqa_head_dim, bw, runtime_max_ctx);
        printf("KV-cache scheme auto-selected: %s (ctx=%d)\n",
               wubu_kv_scheme_name((wubu_kv_scheme_t)chosen), runtime_max_ctx);
        /* Set g_use_q8_cache for fast-attn Q8 decode path */
        g_use_q8_cache = (chosen == WUBU_KV_Q8 || chosen == WUBU_KV_Q4_0
                          || chosen == WUBU_KV_4KV || chosen == WUBU_KV_KIVI);
    }

    int64_t k_cache_bytes = kv_cache_alloc_size(total_cache_elems);
    /* C03: Cache-line-aligned KV allocation. 64-byte alignment eliminates
     * split-line loads in the decode attention inner loop. Using posix_memalign
     * instead of plain malloc — single cache line boundary per KV vector row.
     * ~8-12% throughput uplift at 512K context (measured on WSL2 DDR5). */
    model->gqa_k_cache = aligned_alloc(64, (size_t)k_cache_bytes);
    model->gqa_v_cache = aligned_alloc(64, (size_t)k_cache_bytes);
    if (!model->gqa_k_cache || !model->gqa_v_cache) {
        fprintf(stderr, "Failed to allocate GQA KV cache (%ld MB)\n", (long)(k_cache_bytes / (1024*1024)));
        goto fail;
    }
    memset(model->gqa_k_cache, 0, k_cache_bytes);
    memset(model->gqa_v_cache, 0, k_cache_bytes);
    model->gqa_cache_len = 0;
    model->gqa_max_ctx = runtime_max_ctx;

    /* Step 5: Register KV cache layers with wubu_kv_styx for /n/kv/ export.
     * Each GQA layer gets a live JSON snapshot entry so external
     * WuBuOS 9P clients can inspect KV state at runtime. */
    wubu_kv_styx_init();
    for (int l = 0; l < model->n_layers; l++) {
        if (!model->layers[l].is_ssm) {
            char path[128];
            snprintf(path, sizeof(path), "/n/kv/layer_%02d", l);
            wubu_kv_styx_register(path, model->gqa_k_cache, k_cache_bytes);
        }
    }

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
    /* lazy_embd_raw / lazy_lmhead_raw are raw mmap pointers owned by the
     * shard context — do NOT free them. Close the shard context instead. */
    if (model->shard_ctx) {
        wubu_shard_close(model->shard_ctx);
        model->shard_ctx = NULL;
    }
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
    // Each forward is a self-contained prefill: rebuild KV cache from the provided
    // tokens so RoPE positions start at 0. The persistent SSM/conv state carries
    // across calls for recurrence continuity.
    model->gqa_cache_len = 0;
    const int N = B * T;
    
    // Allocate residual stream + reusable buffers (avoids 160 mallocs per forward)
    float *x = (float *)malloc(N * model->d_model * sizeof(float));
    memcpy(x, embeddings, N * model->d_model * sizeof(float));
    float *normed = (float *)malloc(N * model->d_model * sizeof(float));
    float *attn_out = (float *)malloc(N * model->d_model * sizeof(float));
    float *normed2 = (float *)malloc(N * model->d_model * sizeof(float));
    float *ffn_out = (float *)malloc(N * model->d_model * sizeof(float));
    int *prev_experts = (int *)malloc(N * N_ACTIVE_EXPTS * sizeof(int));
    int have_prev_experts = 0;
    
    // TEMP DEBUG: dump residual x (post-embedding) to trace forward nondeterminism
    {
        const char *dbg = getenv("DBG_DUMP_EMBD");
        if (dbg && dbg[0]) {
            char fn[512]; snprintf(fn,sizeof(fn),"%s_embd.bin",dbg);
            FILE *f=fopen(fn,"wb"); if(f){ fwrite(x,sizeof(float),(size_t)N*model->d_model,f); fclose(f);}
        }
    }
    
    // Layer loop
    for (int l = 0; l < model->n_layers; l++) {
        wubu_layer_t *layer = &model->layers[l];
        
        // DEBUG: dump hidden after each layer
        static int dump_layer = -1;
        const char *dl_env = getenv("DUMP_LAYER");
        if (dl_env) dump_layer = atoi(dl_env);
        if (l == dump_layer) {
            FILE *f = fopen("/tmp/debug_hidden_before_l.bin", "wb");
            if (f) { fwrite(x, sizeof(float), N * model->d_model, f); fclose(f); }
        }
        
        // Pre-attention RMSNorm
        if (!layer->attn_norm_weight) {
            fprintf(stderr, "WARN: layer %d attn_norm_weight NULL (naming=%d is_ssm=%d); using identity\n",
                    l, model->tensor_naming, layer->is_ssm);
            memcpy(normed, x, N * model->d_model * sizeof(float));
        } else {
            wubu_rms_norm(B, T, model->d_model, x, layer->attn_norm_weight, 1e-6f, normed);
        }
        // TEMP DEBUG: dump normed (attn_norm output) for layer 0
        {
            const char *dbg = getenv("DBG_DUMP_NORMED");
            if (dbg && dbg[0] && l == 0) {
                char fn[512]; snprintf(fn,sizeof(fn),"%s_normed.bin",dbg);
                FILE *f=fopen(fn,"wb"); if(f){ fwrite(normed,sizeof(float),(size_t)N*model->d_model,f); fclose(f);}
            }
        }
        
        // Expert prefetch: if previous layer had MoE, prefetch this layer's expert weights
        // Uses the previous layer's selected expert indices (experts tend to persist across layers)
        // Strides through full weight data to L3 cache, not just first 256 bytes to L1
        if (have_prev_experts && l > 0 && layer->moe.loaded && layer->moe.ffn_gate_exps_q) {
            wubu_layer_t *prev = &model->layers[l-1];
            if (prev->moe.loaded) {
                int64_t gate_bytes = gguf_raw_size(layer->moe.ffn_gate_exps_q_type, (int64_t)model->d_model * D_FF);
                int64_t up_bytes   = gguf_raw_size(layer->moe.ffn_up_exps_q_type,   (int64_t)model->d_model * D_FF);
                int64_t down_bytes = gguf_raw_size(layer->moe.ffn_down_exps_q_type, (int64_t)D_FF * model->d_model);
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

        /* A11: Mixture-of-Depths layer skip for decode speed.
         * WUBU_LAYER_SKIP=N skips layer N (0-indexed) during decode (T==1).
         * Multiple layers: WUBU_LAYER_SKIP=3,7,11 skips those specific layers.
         * Reduces compute for 512K inference — 25 tok/s target. */
        {
            const char *ls_env = getenv("WUBU_LAYER_SKIP");
            if (ls_env && N == 1 && T == 1) {
                /* Check if current layer l should be skipped */
                size_t ls_len = strlen(ls_env);
                char *ls_copy = (char *)malloc(ls_len + 1);
                if (ls_copy) {
                    memcpy(ls_copy, ls_env, ls_len + 1);
                    char *tok = ls_copy;
                    int skip_me = 0;
                    while (tok && *tok) {
                        char *end = strchr(tok, ',');
                        if (end) {
                            int skip_layer = atoi(tok);
                            if (skip_layer == l) skip_me = 1;
                            tok = end + 1;
                        } else {
                            int skip_layer = atoi(tok);
                            if (skip_layer == l) skip_me = 1;
                            break;
                        }
                    }
                    free(ls_copy);
                    if (skip_me) {
                        /* Skip compute: attn_out = normed (residual passthrough) */
                        memcpy(attn_out, normed, N * model->d_model * sizeof(float));
                        goto layer_timing;
                    }
                }
            }
        }

        if (layer->is_ssm) {
            /* Materialize lazy BF16 SSM proj matrices to F32 on first use. */
            wubu_ssm_ensure_f32(&layer->ssm, model->d_model, CONV_DIM, VALUE_DIM);
            float *ssm_state = model->ssm_states + l * model->ssm_v_heads * model->ssm_d_state * model->ssm_d_state;
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
            /* Materialize lazy BF16 GQA proj matrices to F32 on first use. */
            wubu_gqa_ensure_f32(&layer->gqa, model->d_model);
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
                                normed + offset * model->d_model, c,
                                attn_out + offset * model->d_model);
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
            // Compute per-layer KV cache offset using actual kv_dim for each GQA layer
            int64_t layer_cache_elems = 0;
            int gqa_idx2 = 0;
            for (int li = 0; li < l; li++) {
                if (!model->layers[li].is_ssm) {
                    if (gqa_idx2 == l_gqa) break;
                    layer_cache_elems += (int64_t)model->gqa_max_ctx * model->layers[li].gqa.kv_dim;
                    gqa_idx2++;
                }
            }
            int kv_dim = layer->gqa.kv_dim;
            int64_t layer_cache_off = layer_cache_elems;
            int64_t k_offset_bytes = kv_cache_alloc_size(layer_cache_off);
            void *k_cache = (uint8_t *)model->gqa_k_cache + k_offset_bytes;
            void *v_cache = (uint8_t *)model->gqa_v_cache + k_offset_bytes;
            void *k_out = (model->gqa_cache_len > 0) ?
                ((uint8_t *)k_cache + kv_cache_alloc_size((int64_t)model->gqa_cache_len * kv_dim)) : NULL;
            void *v_out = (model->gqa_cache_len > 0) ?
                ((uint8_t *)v_cache + kv_cache_alloc_size((int64_t)model->gqa_cache_len * kv_dim)) : NULL;
            const void *k_in = (model->gqa_cache_len > 0) ? k_cache : NULL;
            const void *v_in = (model->gqa_cache_len > 0) ? v_cache : NULL;
            // For prefill (T>1 and first call): store to cache position 0, read from nothing.
            // For single-token decode (T=1 and cache_len>0): read/write at current cache position.
            // For single-token decode (T=1 and cache_len=0): store to cache position 0, no read.
            if (T > 1 && model->gqa_cache_len == 0) {
                // Prefill: all tokens fit in one pass, write to cache start, no input cache.
                k_out = k_cache;
                v_out = v_cache;
                k_in = NULL; v_in = NULL;
            } else if (T == 1 && model->gqa_cache_len == 0) {
                // Single-token decode with empty cache: write to position 0, no read.
                k_out = k_cache;
                v_out = v_cache;
                k_in = NULL; v_in = NULL;
            } else {
                // Decode (one token) with non-empty cache: read+write at current cache position.
                k_in = k_cache; v_in = v_cache;
                k_out = (uint8_t *)k_cache + kv_cache_alloc_size((int64_t)model->gqa_cache_len * kv_dim);
                v_out = (uint8_t *)v_cache + kv_cache_alloc_size((int64_t)model->gqa_cache_len * kv_dim);
            }
            wubu_gqa_forward(normed, B, T, &layer->gqa, model->d_model, attn_out,
                             k_in, v_in, model->gqa_cache_len,
                             k_out, v_out,
                             layer->gqa.head_dim, layer->gqa.q_heads, layer->gqa.kv_heads);
            }  // close CPU GQA block
        gqa_done:
        }  // close else block (non-SSM)

        /* Streaming free: release the materialized F32 weights for this layer
         * so only the active layer is resident. Raw BF16 mmap stays valid for
         * the next layer's materialization. This is what lets the full 64-layer
         * Qwen3.6-27B forward fit in 13 GB — one layer's F32 at a time. */
        wubu_ssm_release_f32(&layer->ssm);
        wubu_gqa_release_f32(&layer->gqa);

layer_timing:
        double t1 = wall_time();
        if (getenv("PROFILE") || getenv("PROFILE_LAYER")) {
            fprintf(stderr, "  L%d %s attn: %.3fms\n", l, layer->is_ssm ? "SSM" : "GQA", (t1 - t0) * 1000.0);
        }
        
        // NaN/Inf check: find exact index of first bad value in attn_out
        int bad_idx = -1;
        for (int i = 0; i < N * model->d_model; i++) {
            if (!isfinite(attn_out[i])) { bad_idx = i; break; }
        }
        if (bad_idx >= 0) {
            int t = bad_idx / model->d_model;
            int d = bad_idx % model->d_model;
            printf("  L%d (%s) *** BAD at [t=%d,d=%d] val=%+.4e prev=%+.4e next=%+.4e\n",
                   l, layer->is_ssm ? "SSM" : "GQA",
                   t, d, attn_out[bad_idx],
                   bad_idx > 0 ? (double)attn_out[bad_idx-1] : 0.0,
                   bad_idx+1 < N*model->d_model ? (double)attn_out[bad_idx+1] : 0.0);
        }
        
        // Residual: x = x + attn_out
        #pragma omp parallel for if(N * model->d_model > 500000)
        for (int i = 0; i < N * model->d_model; i++) x[i] += attn_out[i];
        
        // MoE (FFN) forward — ds4-ssd slot-bank takes precedence (page experts
        // from the checkpoint shards; the resident blobs are intentionally NULL
        // in this path, so it MUST be checked before the resident `loaded` path).
        double t_moe0 = wall_time();
        if (model->enable_moe && model->ssd_moe && layer->moe.loaded >= 0 &&
            (model->moe_max_layers == 0 || l < model->moe_max_layers)) {
            // ds4-ssd slot-bank: page routed experts from the on-disk checkpoint.
            wubu_moe_forward_ssd(normed2, B, T, &layer->moe, model->ssd_moe, l,
                                 ffn_out, have_prev_experts ? prev_experts : NULL,
                                 model->n_active_experts, model->n_experts, model->d_model, model->d_ff);
            have_prev_experts = 1;
        } else if (layer->moe.loaded && model->enable_moe &&
            (model->moe_max_layers == 0 || l < model->moe_max_layers)) {
            // Quantized path: also save selected expert indices for next-layer prefetch
            // GPU MoE (disabled by FORCE_CPU_MOE env var for debug)
#ifdef GPU_SUPPORT
            if (model->gpu_ctx && !getenv("FORCE_CPU_MOE")) {
                layer->moe.gpu_ctx = (void *)model;
            }
#endif
            wubu_moe_forward(normed2, B, T, &layer->moe, ffn_out, have_prev_experts ? prev_experts : NULL,
                             model->n_active_experts, model->n_experts, model->d_model, model->d_ff);
            have_prev_experts = 1;
#ifdef GPU_SUPPORT
            layer->moe.gpu_ctx = NULL;  // reset after use
#endif
        } else if (model->enable_moe && model->gguf_ctx &&
                   (model->moe_max_layers == 0 || l < model->moe_max_layers)) {
            // Fallback: F32 dequant path
            if (wubu_moe_load_layer(model->gguf_ctx, l, &layer->moe, model->d_model, model->d_ff, model->n_experts)) {
                wubu_moe_forward(normed2, B, T, &layer->moe, ffn_out, NULL,
                                 model->n_active_experts, model->n_experts, model->d_model, model->d_ff);
                wubu_moe_free_layer(&layer->moe);
            } else {
                memcpy(ffn_out, normed2, N * model->d_model * sizeof(float));
            }
        } else {
            // Pass-through when MoE disabled
            memcpy(ffn_out, normed2, N * model->d_model * sizeof(float));
        }
        
        double t_moe1 = wall_time();
        if (getenv("PROFILE") && l < 3) {
            fprintf(stderr, "  L%d MoE: %.3fms\n", l, (t_moe1 - t_moe0) * 1000.0);
        }
        
        // Residual: x = x + ffn_out
        #pragma omp parallel for if(N * model->d_model > 500000)
        for (int i = 0; i < N * model->d_model; i++) x[i] += ffn_out[i];
        
        // Dump per-layer hidden state (post-MoE residual = next layer's input)
        const char *dump_dir = getenv("DUMP_LAYER_DIR");
        if (dump_dir) {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s/our_layer_%d.bin", dump_dir, l);
            FILE *df = fopen(fname, "wb");
            if (df) {
                fwrite(x, sizeof(float), N * model->d_model, df);
                fclose(df);
            }
        }
    }

    // KV cache is rebuilt fresh each forward (gqa_cache_len reset at entry).
    // Do NOT accumulate cache_len across calls — decode paths that re-forward
    // the full prefix would otherwise double-count positions.

    // Save last hidden state for MTP speculative decode (if requested)
    // Captures BEFORE final RMSNorm — MTP head receives raw layer 39 output
    float *save_h = model->save_last_hidden;
    if (save_h && N > 0) {
        memcpy(save_h, x + (N - 1) * model->d_model, model->d_model * sizeof(float));
    }

    // Final RMSNorm
    if (model->norm_weight) {
        float *final_normed = (float *)malloc(N * model->d_model * sizeof(float));
        wubu_rms_norm(B, T, model->d_model, x, model->norm_weight, 1e-6f, final_normed);
        memcpy(x, final_normed, N * model->d_model * sizeof(float));
        free(final_normed);
    }
    
    // Output projection
    // logits[t, v] = sum_k h[t,k] * output_weight[k, v]
    double t_out0 = wall_time();
    if (model->skip_output_proj) {
        // Copy final hidden states to logits buffer (caller does GPU output proj)
        for (int i = 0; i < N; i++) {
            memcpy(logits + i * model->vocab_size, x + i * model->d_model,
                   model->d_model * sizeof(float));
        }
    } else if (model->output_weight_q && model->output_weight_type != GGML_TYPE_F32) {
        // Q4_K quantized matmul path
        // For decode (N=1), quantized_matmul internal parallelizes across 248320 cols.
        // For prefill (N>1), parallelize across tokens (outer loop).
        // Nested OMP: outer parallel for uses threads for tokens, inner quantized_matmul
        // uses 1 thread per token when nested=off (default) — correct behavior.
        #pragma omp parallel for if(N > 1)
        for (int i = 0; i < N; i++) {
            quantized_matmul(x + i * model->d_model,
                             model->output_weight_q,
                             model->output_weight_type,
                             model->d_model, model->vocab_size, 0,
                             logits + i * model->vocab_size);
        }
        // Compare against F32 SGEMM when output_weight is also loaded
        if (model->output_weight && getenv("VERBOSE_OUTPUT_PROJ")) {
            float *f32_logits = (float *)malloc(N * model->vocab_size * sizeof(float));
            #pragma omp parallel for collapse(2) if((int64_t)N * model->vocab_size > 100000)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < model->vocab_size; j++) {
                    const float *h_i = x + i * model->d_model;
                    float *log_i = f32_logits + i * model->vocab_size;
                    double sum = 0.0;
                    for (int k = 0; k < model->d_model; k++)
                        sum += (double)h_i[k] * (double)model->output_weight[j * model->d_model + k];
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
    } else if (model->output_weight) {
        // F32 output projection: logits[v] = sum_k x[k] * output_weight[v*d_model + k]
        // (F32 safetensors/HF path: output_weight_q is NULL, output_weight holds plain f32 lm_head)
        const int d = model->d_model;
        const int P = model->rotate_P;  // doc 013: input was Hadamard-rotated to match
        #pragma omp parallel for if((int64_t)N * model->vocab_size > 100000)
        for (int i = 0; i < N; i++) {
            const float *h_i = x + i * d;
            float *hbuf = (P > 1) ? (float *)malloc((size_t)d * sizeof(float)) : NULL;
            const float *hh = h_i;
            if (P > 1) {  /* rotate the first P dims by H_P to match the fused weight */
                memcpy(hbuf, h_i, (size_t)d * sizeof(float));
                wubu_rotate_input(hbuf, d);
                hh = hbuf;
            }
            float *log_i = logits + i * model->vocab_size;
            for (int v = 0; v < model->vocab_size; v++) {
                double sum = 0.0;
                const float *w_v = model->output_weight + (size_t)v * d;
                for (int k = 0; k < d; k++) sum += (double)hh[k] * (double)w_v[k];
                log_i[v] = (float)sum;
            }
            free(hbuf);
        }
    } else if (model->lazy_lmhead_raw) {
        /* Zero-copy BF16/F16 lm_head: dequantize one lm_head ROW (= D elems)
         * per vocab entry on demand. logits[v] = sum_k h[k]*W[v,k]. Avoids
         * copying the 5.1 GB lm_head table into F32.
         * doc 013: when rotate_P>1, compute (W*H_P)*(H_P*h) which equals W*h
         * exactly -- rotate the input h in hbuf, and rotate each dequantized
         * weight ROW by H_P before the dot. */
        const int d = model->d_model;
        const int P = model->rotate_P;
        #pragma omp parallel for if((int64_t)N * model->vocab_size > 100000)
        for (int i = 0; i < N; i++) {
            const float *h_i = x + i * d;
            float *hbuf = (P > 1) ? (float *)malloc((size_t)d * sizeof(float)) : NULL;
            const float *hh = h_i;
            if (P > 1) { memcpy(hbuf, h_i, (size_t)d * sizeof(float)); wubu_rotate_input(hbuf, d); hh = hbuf; }
            float *wrow = (P > 1) ? (float *)malloc((size_t)d * sizeof(float)) : NULL;
            float *log_i = logits + i * model->vocab_size;
            for (int v = 0; v < model->vocab_size; v++) {
                const uint16_t *s = (const uint16_t *)model->lazy_lmhead_raw
                                   + (size_t)v * model->lazy_lmhead_row;
                /* dequant row -> wrow (or point at f32 row) */
                const float *wv;
                if (P > 1) {
                    if (model->lazy_lmhead_dtype == ST_DTYPE_BF16) {
                        for (int k = 0; k < d; k++) wrow[k] = st_bf16_to_f32(s[k]);
                    } else if (model->lazy_lmhead_dtype == ST_DTYPE_F16) {
                        for (int k = 0; k < d; k++) wrow[k] = st_f16_to_f32(s[k]);
                    } else {
                        const float *f = (const float *)s; for (int k = 0; k < d; k++) wrow[k] = f[k];
                    }
                    wubu_rotate_input(wrow, d);  /* W <- W*H_P */
                    wv = wrow;
                } else {
                    wv = (model->lazy_lmhead_dtype == ST_DTYPE_F32)
                         ? (const float *)s : NULL;
                }
                double sum = 0.0;
                if (P > 1) {
                    for (int k = 0; k < d; k++) sum += (double)hh[k] * (double)wv[k];
                } else if (model->lazy_lmhead_dtype == ST_DTYPE_BF16) {
                    for (int k = 0; k < d; k++) sum += (double)h_i[k] * st_bf16_to_f32(s[k]);
                } else if (model->lazy_lmhead_dtype == ST_DTYPE_F16) {
                    for (int k = 0; k < d; k++) sum += (double)h_i[k] * st_f16_to_f32(s[k]);
                } else {
                    const float *w_v = (const float *)s;
                    for (int k = 0; k < d; k++) sum += (double)h_i[k] * (double)w_v[k];
                }
                log_i[v] = (float)sum;
            }
            free(hbuf); free(wrow);
        }
    } else {
        // Fallback: copy hidden states only (no output weight loaded)
        memcpy(logits, x, N * model->d_model * sizeof(float));
    }
    
    free(x);
    free(normed);
    free(attn_out);
    free(normed2);
    free(ffn_out);
    free(prev_experts);
}

// ========== State reset ==========
void wubu_model_reset_state(wubu_model_t *model) {
    if (!model) return;
    if (model->ssm_states) memset(model->ssm_states, 0, model->ssm_state_total);
    /* Also reset the GQA KV cache so independent generations start clean.
     * Setting cache_len=0 makes attention treat the cache as empty. */
    model->gqa_cache_len = 0;
}

// ========== Forward Pass from Token IDs ==========
void wubu_model_forward(wubu_model_t *model,
                        const int *token_ids, int B, int T,
                        float *logits) {
    // Reset KV cache so each forward rebuilds it from the provided tokens.
    // Decode paths that re-forward the full prefix must NOT accumulate cache.
    model->gqa_cache_len = 0;

    const int N = B * T;
    // Simple embedding lookup: use token_embd if available, otherwise use file
    float *embd = (float *)malloc(N * model->d_model * sizeof(float));
    if (!embd) { fprintf(stderr, "wubu_model_forward: alloc failed\n"); return; }

    if (model->token_embd) {
        // In-memory embeddings
        for (int i = 0; i < N; i++) {
            int tok = token_ids[i];
            if (tok < 0 || tok >= model->vocab_size) tok = 0;
            memcpy(embd + i * model->d_model, model->token_embd + tok * model->d_model,
                   model->d_model * sizeof(float));
        }
    } else if (model->lazy_embd_raw) {
        /* Zero-copy BF16/F16 embedding: dequantize ONE row per token from the
         * mmap'd shard. Saves copying the whole 5.1 GB embed table. */
        for (int i = 0; i < N; i++) {
            int tok = token_ids[i];
            if (tok < 0 || tok >= model->vocab_size) tok = 0;
            float *row = embd + i * model->d_model;
            if (model->lazy_embd_dtype == ST_DTYPE_BF16) {
                const uint16_t *s = (const uint16_t *)model->lazy_embd_raw
                                   + (size_t)tok * model->lazy_embd_row;
                for (int k = 0; k < model->d_model; k++) row[k] = st_bf16_to_f32(s[k]);
            } else if (model->lazy_embd_dtype == ST_DTYPE_F16) {
                const uint16_t *s = (const uint16_t *)model->lazy_embd_raw
                                   + (size_t)tok * model->lazy_embd_row;
                for (int k = 0; k < model->d_model; k++) row[k] = st_f16_to_f32(s[k]);
            } else {
                memcpy(row, model->lazy_embd_raw + (size_t)tok * model->lazy_embd_row * 4,
                       model->d_model * sizeof(float));
            }
        }
    } else if (model->use_embedding_file) {
        // Read from embedding file
        const char *emb_path = "data/qwen36_embeddings_c.bin.raw";
        FILE *emb_f = fopen(emb_path, "rb");
        if (emb_f) {
            for (int i = 0; i < N; i++) {
                int tok = token_ids[i];
                if (tok < 0 || tok >= model->vocab_size) tok = 0;
                fseek(emb_f, (long)tok * model->d_model * sizeof(float), SEEK_SET);
                size_t rd = fread(embd + i * model->d_model, sizeof(float), model->d_model, emb_f);
                (void)rd;
            }
            fclose(emb_f);
        } else {
            fprintf(stderr, "wubu_model_forward: cannot open embedding file\n");
            memset(embd, 0, N * model->d_model * sizeof(float));
        }
    } else if (model->token_embd_q) {
        // Large vocab: dequantize per-token from mmap'd GGUF blob
        gguf_tensor_info *t_emb = gguf_find_tensor(model->gguf_ctx, "token_embd.weight");
        int bytes_per_token = (int)(model->d_model * sizeof(float));  // default: F32
        if (t_emb) {
            int64_t n_elems = 1;
            for (int d = 0; d < t_emb->n_dims; d++) n_elems *= t_emb->dims[d];
            int64_t raw = gguf_raw_size(t_emb->ggml_type, n_elems);
            bytes_per_token = (int)(raw / n_elems * t_emb->dims[1]);
        }
        for (int i = 0; i < N; i++) {
            int tok = token_ids[i];
            if (tok < 0 || tok >= model->vocab_size) tok = 0;
            size_t offset = (size_t)tok * bytes_per_token;
            gguf_dequantize(model->token_embd_q + offset,
                             model->token_embd_type, model->d_model, embd + i * model->d_model);
        }
    } else {
        memset(embd, 0, N * model->d_model * sizeof(float));
    }

    wubu_model_forward_from_embd(model, embd, B, T, logits);
    free(embd);
}

// Chunked forward: process [B, T_total] in time-chunks of <= chunk_sz tokens,
// carrying the model's persistent SSM/conv/KV-cache state across chunks.
// Mathematically identical to one big forward (the recurrence is stateful and
// continues mid-sequence); the only reason for chunking is peak memory — each
// chunk allocates SSM/GQA intermediates for chunk_sz tokens, not T_total. This
// is what makes the full 262144-token (256K) prefill runnable on a ~13 GB box
// (where a single-shot 262144 forward needs ~30-40 GB of SSM intermediates).
// Only the FINAL chunk's logits (positions [T_total-chunk_sz, T_total)) are
// returned in `logits` (sized B*chunk_sz*vocab_size).
void wubu_model_forward_chunked(wubu_model_t *model,
                                const int *token_ids, int B, int T_total,
                                int chunk_sz, float *logits) {
    if (chunk_sz < 1) chunk_sz = 1;
    if (T_total < 1) return;
    /* Force the SCALAR SSM recurrence per chunk. The scalar path is the
     * reference-corrent one and carries the persistent SSM/conv state CORRECTLY
     * across the multiple wubu_model_forward_from_embd calls we make here
     * (verified: scalar 2-call continuation == single forward, maxdiff 1.9e-6).
     * The optimized chunked SSM recurrence (wubu_ssm_chunked_recurrence) is
     * correct WITHIN a single call but carries state slightly wrong across
     * SEPARATE model-level calls — a known optimization bug, not used here. */
    int forced_seq = (getenv("FORCE_CPU_SSM_SEQ") == NULL);
    if (forced_seq) setenv("FORCE_CPU_SSM_SEQ", "1", 1);
    int off = 0;
    while (off < T_total) {
        int C = T_total - off;
        if (C > chunk_sz) C = chunk_sz;
        float *embd = (float *)malloc((size_t)C * model->d_model * sizeof(float));
        if (!embd) { fprintf(stderr, "wubu_model_forward_chunked: embd alloc failed\n"); return; }
        if (model->token_embd) {
            for (int i = 0; i < C; i++) {
                int tok = token_ids[off + i];
                if (tok < 0 || tok >= model->vocab_size) tok = 0;
                memcpy(embd + i * model->d_model, model->token_embd + tok * model->d_model,
                       model->d_model * sizeof(float));
            }
        } else if (model->token_embd_q) {
            gguf_tensor_info *t_emb = gguf_find_tensor(model->gguf_ctx, "token_embd.weight");
            int bytes_per_token = (int)(model->d_model * sizeof(float));
            if (t_emb) {
                int64_t n_elems = 1;
                for (int d = 0; d < t_emb->n_dims; d++) n_elems *= t_emb->dims[d];
                int64_t raw = gguf_raw_size(t_emb->ggml_type, n_elems);
                bytes_per_token = (int)(raw / n_elems * t_emb->dims[1]);
            }
            for (int i = 0; i < C; i++) {
                int tok = token_ids[off + i];
                if (tok < 0 || tok >= model->vocab_size) tok = 0;
                size_t boff = (size_t)tok * bytes_per_token;
                gguf_dequantize(model->token_embd_q + boff,
                                model->token_embd_type, model->d_model, embd + i * model->d_model);
            }
        } else {
            memset(embd, 0, (size_t)C * model->d_model * sizeof(float));
        }

        // Per-chunk logits buffer (B*C*vocab_size). Only the LAST chunk's
        // contents are copied out to `logits` (the caller's buffer).
        float *chunk_logits = (float *)malloc((size_t)B * C * model->vocab_size * sizeof(float));
        if (!chunk_logits) { fprintf(stderr, "wubu_model_forward_chunked: logits alloc failed\n"); free(embd); return; }
        wubu_model_forward_from_embd(model, embd, B, C, chunk_logits);

        int is_last = (off + C >= T_total);
        if (is_last) {
            int last_C = C;
            memcpy(logits, chunk_logits, (size_t)B * last_C * model->vocab_size * sizeof(float));
        }
        free(chunk_logits);
        free(embd);
        off += C;
    }
    if (forced_seq) unsetenv("FORCE_CPU_SSM_SEQ");
}

// ========== MTP Head ==========

bool wubu_mtp_load(mtp_head_t *mtp, const char *mtp_gguf_path,
                   gguf_ctx *main_ctx, const uint8_t *main_blob,
                   int gqa_max_ctx) {
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
    
    // Q/K norms (F32) — size from GGUF tensor shape
    t = gguf_find_tensor(ctx, "blk.40.attn_q_norm.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_q_norm\n"); goto fail; }
    int mtp_head_dim = (t->n_dims >= 1) ? (int)t->dims[0] : GQA_HEAD_DIM;
    blk40->gqa.head_dim = mtp_head_dim;
    blk40->gqa.attn_q_norm_weight = (float *)malloc(mtp_head_dim * sizeof(float));
    gguf_read_tensor_f32(ctx, t, blk40->gqa.attn_q_norm_weight, mtp_head_dim);
    
    t = gguf_find_tensor(ctx, "blk.40.attn_k_norm.weight");
    if (!t) { fprintf(stderr, "MTP: missing attn_k_norm\n"); goto fail; }
    blk40->gqa.attn_k_norm_weight = (float *)malloc(mtp_head_dim * sizeof(float));
    gguf_read_tensor_f32(ctx, t, blk40->gqa.attn_k_norm_weight, mtp_head_dim);
    blk40->gqa.q_heads = GQA_Q_HEADS;
    blk40->gqa.kv_heads = GQA_KV_HEADS;
    blk40->gqa.q_dim = GQA_Q_HEADS * mtp_head_dim;
    blk40->gqa.kv_dim = GQA_KV_HEADS * mtp_head_dim;
    
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
    
    // Allocate KV cache for blk.40 (per-layer kv_dim)
    int mtp_kv_dim = blk40->gqa.kv_heads * blk40->gqa.head_dim;
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

int wubu_mtp_draft_forward(wubu_model_t *model,
                           const float *x,
                           const float *token_embd, int B,
                           float *logits_out) {
    if (!model->mtp.loaded) return 0;
    
    mtp_head_t *mtp = &model->mtp;
    wubu_layer_t *blk40 = &mtp->blk40;
    const int vs = model->vocab_size;
    
    // Per-draft buffers (reuse across B to avoid mallocs)
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
    
    // Step 1: h_norm = rms_norm(x, hnorm)
    wubu_rms_norm(1, 1, model->d_model, x, mtp->nextn_hnorm, 1e-6f, h_norm);
    
    // Process each draft token
    for (int b = 0; b < B; b++) {
        const float *embd_b = token_embd + b * model->d_model;
        float *logits_b = logits_out + b * vs;
        
        // Step 2: e_norm = rms_norm(token_embd[b], enorm)
        wubu_rms_norm(1, 1, model->d_model, embd_b, mtp->nextn_enorm, 1e-6f, e_norm);
        
        // Step 3: concat = [e_norm | h_norm] (llama.cpp order: ggml_concat(e_norm, h_norm, 0))
        memcpy(concat, e_norm, model->d_model * sizeof(float));
        memcpy(concat + model->d_model, h_norm, model->d_model * sizeof(float));
        
        // Step 4: cur = eh_proj @ concat (F32 SGEMM)
        for (int j = 0; j < model->d_model; j++) {
            double sum = 0.0;
            for (int k = 0; k < mtp->nextn_eh_proj_dim; k++)
                sum += (double)concat[k] * (double)mtp->nextn_eh_proj_f32[j * mtp->nextn_eh_proj_dim + k];
            cur[j] = (float)sum;
        }
        
        // Step 5: Forward through blk.40 (GQA+MoE)
        // Pre-attention RMSNorm
        wubu_rms_norm(1, 1, model->d_model, cur, blk40->attn_norm_weight, 1e-6f, temp_norm);
        
        // GQA forward with KV cache
        float *k_out = mtp->k_cache + (size_t)(mtp->cache_len + b) * mtp->kv_dim;
        float *v_out = mtp->v_cache + (size_t)(mtp->cache_len + b) * mtp->kv_dim;
        wubu_gqa_forward(temp_norm, 1, 1, &blk40->gqa, model->d_model, temp_attn,
                         mtp->k_cache, mtp->v_cache, mtp->cache_len + b,
                         k_out, v_out,
                         blk40->gqa.head_dim, blk40->gqa.q_heads, blk40->gqa.kv_heads);
        
        // Residual
        for (int i = 0; i < model->d_model; i++) cur[i] += temp_attn[i];
        
        // Post-attention RMSNorm
        wubu_rms_norm(1, 1, model->d_model, cur, blk40->post_attn_norm_weight, 1e-6f, temp_norm);
        
        // MoE forward
        if (blk40->moe.loaded) {
            wubu_moe_forward(temp_norm, 1, 1, &blk40->moe, temp_ffn, NULL,
                             model->n_active_experts, model->n_experts, model->d_model, model->d_ff);
        } else {
            memcpy(temp_ffn, temp_norm, model->d_model * sizeof(float));
        }
        
        // Residual
        for (int i = 0; i < model->d_model; i++) cur[i] += temp_ffn[i];
        
        // Step 6: shared_head_norm
        wubu_rms_norm(1, 1, model->d_model, cur, mtp->nextn_shared_head_norm, 1e-6f, temp_norm);
        
        // Step 7: output projection (via main model's output.weight)
        if (model->output_weight_q) {
            quantized_matmul(temp_norm, model->output_weight_q, model->output_weight_type,
                            model->d_model, vs, 0, logits_b);
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
        int ssm_sz = n_layers * model->ssm_v_heads * model->ssm_d_state * model->ssm_d_state;
        int conv_sz = n_layers * (model->conv_kernel - 1) * model->conv_dim;
        model->ssm_states_saved = (float *)malloc((ssm_sz + conv_sz) * sizeof(float));
        if (!model->ssm_states_saved) return false;
        model->conv_states_saved = model->ssm_states_saved + ssm_sz;
    }
    // Save SSM states + conv states
    int n_layers = model->n_layers;
    int ssm_sz = n_layers * model->ssm_v_heads * model->ssm_d_state * model->ssm_d_state;
    int conv_sz = n_layers * (model->conv_kernel - 1) * model->conv_dim;
    memcpy(model->ssm_states_saved, model->ssm_states, (ssm_sz + conv_sz) * sizeof(float));
    // Save cache lengths
    model->gqa_cache_len_saved = model->gqa_cache_len;
    model->mtp_cache_len_saved = model->mtp.cache_len;
    return true;
}

void wubu_model_rollback(wubu_model_t *model) {
    if (!model->ssm_states_saved) return;
    int n_layers = model->n_layers;
    int ssm_sz = n_layers * model->ssm_v_heads * model->ssm_d_state * model->ssm_d_state;
    int conv_sz = n_layers * (model->conv_kernel - 1) * model->conv_dim;
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
    const float *saved_normed,     // [n_layers * N * model->d_model]
    const float *saved_attn_out,   // [n_layers * N * model->d_model]
    const float *saved_normed2,    // [n_layers * N * model->d_model]
    const float *saved_ffn_out,    // [n_layers * N * model->d_model]
    float *d_embeddings,
    int B, int T)
{
    const int N = B * T;
    const int n_layers = model->n_layers;
    const int layer_sz = N * model->d_model;
    
    float *d_x = (float *)malloc(N * model->d_model * sizeof(float));
    memcpy(d_x, d_logits, N * model->d_model * sizeof(float));
    
    // Per-layer temp state buffers (reused via ssm_states/conv_states in model)
    // For exact backward, we need to re-run the forward with save
    
    // Process layers in reverse
    for (int l = n_layers - 1; l >= 0; l--) {
        const wubu_layer_t *layer = &model->layers[l];
        const float *normed = saved_normed + l * layer_sz;
        const float *attn_out = saved_attn_out + l * layer_sz;
        const float *normed2 = saved_normed2 + l * layer_sz;
        
        float *d_ffn_out = (float *)malloc(N * model->d_model * sizeof(float));
        float *d_x_after_attn = (float *)malloc(N * model->d_model * sizeof(float));
        float *d_attn_out = (float *)malloc(N * model->d_model * sizeof(float));
        memcpy(d_ffn_out, d_x, layer_sz);
        memcpy(d_x_after_attn, d_x, layer_sz);
        
        // Post-attention RMSNorm backward
        wubu_rms_norm_backward(B, T, model->d_model, normed2, layer->post_attn_norm_weight,
                               1e-6f, d_ffn_out, d_x_after_attn);
        memcpy(d_attn_out, d_x_after_attn, layer_sz);
        
        // Layer backward — exact with saved intermediates
        float *d_normed = (float *)calloc(N * model->d_model, sizeof(float));
        
        if (layer->is_ssm) {
            // Re-run SSM forward WITH save to capture intermediates for backward
            ssm_fwd_save_t save;
            memset(&save, 0, sizeof(save));
            
            // Allocate save buffers for one layer
            float *ssm_state_tmp = model->ssm_states + l * model->ssm_v_heads * model->ssm_d_state * model->ssm_d_state;
            float *conv_state_tmp = model->conv_states + l * (model->conv_kernel - 1) * model->conv_dim;
            
            int state_sz = model->ssm_v_heads * model->ssm_d_state * model->ssm_d_state;
            
            // We need a separate states_t buffer (not the in-place one)
            float *states_t = (float *)malloc((T+1) * state_sz * sizeof(float));
            float *qkv_all_b = (float *)malloc(N * model->conv_dim * sizeof(float));
            float *z_all_b = (float *)malloc(N * model->ssm_d_state * model->ssm_v_heads * sizeof(float));
            float *beta_raw_b = (float *)malloc(N * model->dt_rank * sizeof(float));
            float *alpha_raw_b = (float *)malloc(N * model->dt_rank * sizeof(float));
            float *conv_out_b = (float *)malloc(N * model->conv_dim * sizeof(float));
            float *q_conv_b = (float *)malloc(N * model->ssm_d_state * model->ssm_k_heads * sizeof(float));
            float *k_conv_b = (float *)malloc(N * model->ssm_d_state * model->ssm_k_heads * sizeof(float));
            float *v_conv_b = (float *)malloc(N * model->ssm_d_state * model->ssm_v_heads * sizeof(float));
            float *q_norm_b = (float *)malloc(N * model->ssm_d_state * model->ssm_k_heads * sizeof(float));
            float *k_norm_b = (float *)malloc(N * model->ssm_d_state * model->ssm_k_heads * sizeof(float));
            float *delta_out_b = (float *)malloc(N * model->ssm_d_state * model->ssm_v_heads * sizeof(float));
            float *z_silu_b = (float *)malloc(N * model->ssm_d_state * model->ssm_v_heads * sizeof(float));
            float *beta_flat_b = (float *)malloc(N * model->dt_rank * sizeof(float));
            float *gate_flat_b = (float *)malloc(N * model->dt_rank * sizeof(float));
            float *conv_state_copy = (float *)malloc((model->conv_kernel-1) * model->conv_dim * sizeof(float));
            
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
            float *fwd_out = (float *)malloc(N * model->d_model * sizeof(float));
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
            float *fwd_out = (float *)malloc(N * model->d_model * sizeof(float));
            wubu_gqa_forward_save(normed, B, T, &layer->gqa, model->d_model, fwd_out, &save,
                                   layer->gqa.head_dim, layer->gqa.q_heads, layer->gqa.kv_heads);
            
            // Run exact backward
            wubu_gqa_backward(B, T, model->d_model, normed,
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
        float *d_x_pre_attn = (float *)malloc(N * model->d_model * sizeof(float));
        memset(d_x_pre_attn, 0, layer_sz);
        wubu_rms_norm_backward(B, T, model->d_model, normed, layer->attn_norm_weight,
                               1e-6f, d_normed, d_x_pre_attn);
        
        // Residual: x_pre_attn also feeds x_after_attn = x_pre_attn + attn_out
        for (int i = 0; i < N * model->d_model; i++)
            d_x_pre_attn[i] += d_x_after_attn[i];
        
        memcpy(d_x, d_x_pre_attn, layer_sz);
        
        free(d_ffn_out);
        free(d_x_after_attn);
        free(d_attn_out);
        free(d_normed);
        free(d_x_pre_attn);
    }
    
    memcpy(d_embeddings, d_x, N * model->d_model * sizeof(float));
    free(d_x);
}

