#include "wubu_gemma4.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ========== Tensor Name Helpers ==========

static void tn_attn_norm(char *buf, int l) { snprintf(buf, 64, "blk.%d.attn_norm.weight", l); }
static void tn_attn_q(char *buf, int l) { snprintf(buf, 64, "blk.%d.attn_q.weight", l); }
static void tn_attn_k(char *buf, int l) { snprintf(buf, 64, "blk.%d.attn_k.weight", l); }
static void tn_attn_v(char *buf, int l) { snprintf(buf, 64, "blk.%d.attn_v.weight", l); }
static void tn_attn_out(char *buf, int l) { snprintf(buf, 64, "blk.%d.attn_output.weight", l); }
static void tn_attn_q_norm(char *buf, int l) { snprintf(buf, 64, "blk.%d.attn_q_norm.weight", l); }
static void tn_attn_k_norm(char *buf, int l) { snprintf(buf, 64, "blk.%d.attn_k_norm.weight", l); }
static void tn_post_attn_norm(char *buf, int l) { snprintf(buf, 64, "blk.%d.post_attention_norm.weight", l); }
static void tn_ffn_norm(char *buf, int l) { snprintf(buf, 64, "blk.%d.ffn_norm.weight", l); }
static void tn_ffn_gate(char *buf, int l) { snprintf(buf, 64, "blk.%d.ffn_gate.weight", l); }
static void tn_ffn_up(char *buf, int l) { snprintf(buf, 64, "blk.%d.ffn_up.weight", l); }
static void tn_ffn_down(char *buf, int l) { snprintf(buf, 64, "blk.%d.ffn_down.weight", l); }
static void tn_post_ffw_norm(char *buf, int l) { snprintf(buf, 64, "blk.%d.post_ffw_norm.weight", l); }
static void tn_out_scale(char *buf, int l) { snprintf(buf, 64, "blk.%d.layer_output_scale.weight", l); }
static void tn_rope_freqs(char *buf, int l) { snprintf(buf, 64, "blk.%d.rope_freqs.weight", l); }

// ========== Helpers ==========

static float *load_tensor_f32(gguf_ctx *ctx, const char *name, int64_t n_elems) {
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) return NULL;
    float *buf = (float *)malloc(n_elems * sizeof(float));
    if (!buf) { fprintf(stderr, "OOM loading %s\n", name); return NULL; }
    if (!gguf_read_tensor_f32(ctx, t, buf, n_elems)) {
        fprintf(stderr, "Failed to read %s\n", name);
        free(buf);
        return NULL;
    }
    printf("  Loaded %s [%lld elems]\n", name, (long long)n_elems);
    return buf;
}

static bool load_layer_tensor_f32(gguf_ctx *ctx, float **dst, const char *name, int64_t n_elems) {
    *dst = load_tensor_f32(ctx, name, n_elems);
    return *dst != NULL;
}

// ========== Init ==========

bool gemma4_init(gemma4_model_t *model, const char *gguf_path) {
    memset(model, 0, sizeof(*model));
    model->n_layers = GEMMA4_N_LAYERS;

    // Open GGUF
    gguf_ctx *ctx = gguf_open(gguf_path);
    if (!ctx) { fprintf(stderr, "Failed to open %s\n", gguf_path); return false; }

    // Buffer data blob for fast access
    if (!gguf_buffer_data(ctx)) {
        fprintf(stderr, "Failed to buffer data blob\n");
        gguf_close(ctx);
        return false;
    }
    model->data_blob = ctx->data_blob;
    model->data_blob_size = ctx->data_blob_size;

    printf("Loading Gemma 4 12B from %s\n", gguf_path);

    // Global tensors
    // token_embd.weight: [3840, 262144]
    int64_t n_embd = GEMMA4_D_MODEL;
    int64_t n_vocab = GEMMA4_VOCAB_SIZE;
    load_layer_tensor_f32(ctx, &model->token_embd_weight, "token_embd.weight", n_embd * n_vocab);

    // output.weight is optional (tied to token_embd if absent)
    model->output_weight = load_tensor_f32(ctx, "output.weight", n_embd * n_vocab);
    if (!model->output_weight) {
        printf("  output.weight not found, tying to token_embd.weight\n");
        model->output_weight = model->token_embd_weight;  // tied
        model->tied_embeddings = true;
    }

    load_layer_tensor_f32(ctx, &model->output_norm_weight, "output_norm.weight", n_embd);
    printf("  Loaded global tensors\n");

    // Allocate layers
    model->layers = (gemma4_layer_t *)calloc(GEMMA4_N_LAYERS, sizeof(gemma4_layer_t));
    if (!model->layers) { gguf_close(ctx); return false; }

    // Per-layer tensors
    for (int i = 0; i < GEMMA4_N_LAYERS; i++) {
        gemma4_layer_t *layer = &model->layers[i];
        layer->layer_idx = i;
        layer->layer_type = g4_layer_is_full(i) ? GEMMA4_LAYER_FULL : GEMMA4_LAYER_SLIDING;
        layer->out_scale = 1.0f;

        char name[64];
        float *buf;

        // Norms
        tn_attn_norm(name, i);
        load_layer_tensor_f32(ctx, &layer->attn_norm_weight, name, n_embd);

        // Q/K norms: head_dim for sliding, global_head_dim for full attention
        int n_head_dim = g4_layer_is_full(i) ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_HEAD_DIM;

        tn_attn_q_norm(name, i);
        load_layer_tensor_f32(ctx, &layer->attn_q_norm_weight, name, n_head_dim);

        tn_attn_k_norm(name, i);
        load_layer_tensor_f32(ctx, &layer->attn_k_norm_weight, name, n_head_dim);

        tn_post_attn_norm(name, i);
        load_layer_tensor_f32(ctx, &layer->post_attention_norm, name, n_embd);

        tn_ffn_norm(name, i);
        load_layer_tensor_f32(ctx, &layer->ffn_norm_weight, name, n_embd);

        tn_post_ffw_norm(name, i);
        load_layer_tensor_f32(ctx, &layer->post_ffw_norm_weight, name, n_embd);

        // Attention projections
        // Q: [3840, n_head * head_dim]
        // Full attention layers use global_head_dim (512) instead of head_dim (256)
        int attn_head_dim = g4_layer_is_full(i) ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_HEAD_DIM;
        int64_t n_q = n_embd * GEMMA4_N_HEADS * attn_head_dim / n_embd;  // 4096 or 8192
        tn_attn_q(name, i);
        load_layer_tensor_f32(ctx, &layer->wq, name, n_embd * n_q);

        // K: [3840, 8*256] = [3840, 2048] — K always uses head_dim, not global (only Q has global)
        // Actually check: full attention still uses n_kv_head * head_dim for K
        int64_t n_k = n_embd * GEMMA4_N_KV_HEADS * GEMMA4_HEAD_DIM / n_embd;  // 2048
        tn_attn_k(name, i);
        load_layer_tensor_f32(ctx, &layer->wk, name, n_embd * n_k);

        // V: optional (falls back to K if absent)
        tn_attn_v(name, i);
        layer->wv = load_tensor_f32(ctx, name, n_embd * n_k);
        if (!layer->wv) {
            printf("  blk.%d.attn_v.weight not found, using K as V\n", i);
            layer->wv = layer->wk;  // K=V fallback
        }

        // Out: [n_head * head_dim, 3840] — same Q-attn-head-count for output
        tn_attn_out(name, i);
        int64_t n_out = GEMMA4_N_HEADS * attn_head_dim;  // 4096 or 8192
        load_layer_tensor_f32(ctx, &layer->wo, name, n_out * n_embd);

        // Layer output scale (optional)
        tn_out_scale(name, i);
        buf = load_tensor_f32(ctx, name, 1);
        if (buf) { layer->out_scale = buf[0]; free(buf); }

        // RoPE freqs (full attention layers only)
        if (g4_layer_is_full(i)) {
            tn_rope_freqs(name, i);
            load_layer_tensor_f32(ctx, &layer->rope_freqs, name, GEMMA4_HEAD_DIM / 2);
        }

        // FFN weights (will be loaded quantized, load as F32 for now)
        int64_t n_ff = GEMMA4_FFN_HIDDEN;

        tn_ffn_gate(name, i);
        load_layer_tensor_f32(ctx, &layer->ffn.gate, name, n_embd * n_ff);

        tn_ffn_up(name, i);
        load_layer_tensor_f32(ctx, &layer->ffn.up, name, n_embd * n_ff);

        tn_ffn_down(name, i);
        load_layer_tensor_f32(ctx, &layer->ffn.down, name, n_ff * n_embd);

        if ((i + 1) % 8 == 0 || i == GEMMA4_N_LAYERS - 1)
            printf("  Layer %d/%d loaded\n", i + 1, GEMMA4_N_LAYERS);
    }

    printf("Model loaded: %d layers, %s embeddings\n",
           GEMMA4_N_LAYERS, model->tied_embeddings ? "tied" : "separate");

    // Allocate FFN workspace
    model->ffn_workspace = (float *)malloc(GEMMA4_FFN_HIDDEN * sizeof(float));

    gguf_close(ctx);
    return true;
}

void gemma4_free(gemma4_model_t *model) {
    if (!model) return;

    // Free data blob
    if (model->data_blob) {
        free(model->data_blob);
    }

    // Global tensors
    if (model->token_embd_weight && !model->tied_embeddings)
        free(model->token_embd_weight);
    if (model->output_weight && model->output_weight != model->token_embd_weight)
        free(model->output_weight);
    free(model->output_norm_weight);

    // Layer tensors
    if (model->layers) {
        for (int i = 0; i < model->n_layers; i++) {
            gemma4_layer_t *l = &model->layers[i];
            free(l->attn_norm_weight);
            free(l->attn_q_norm_weight);
            free(l->attn_k_norm_weight);
            free(l->post_attention_norm);
            free(l->ffn_norm_weight);
            free(l->post_ffw_norm_weight);
            if (l->wq) free(l->wq);
            if (l->wk) free(l->wk);
            if (l->wv && l->wv != l->wk) free(l->wv);
            if (l->wo) free(l->wo);
            if (l->rope_freqs) free(l->rope_freqs);
            if (l->ffn.gate) free(l->ffn.gate);
            if (l->ffn.up) free(l->ffn.up);
            if (l->ffn.down) free(l->ffn.down);
        }
        free(model->layers);
    }

    free(model->ffn_workspace);
    memset(model, 0, sizeof(*model));
}

// ========== KV Cache ==========

bool gemma4_kv_cache_init(gemma4_kv_cache_t *cache, int max_ctx) {
    memset(cache, 0, sizeof(*cache));
    cache->max_ctx = max_ctx;
    cache->n_kv_heads = GEMMA4_N_KV_HEADS;
    cache->head_dim = GEMMA4_HEAD_DIM;

    int64_t cache_size = (int64_t)GEMMA4_N_LAYERS * max_ctx * GEMMA4_N_KV_HEADS * GEMMA4_HEAD_DIM;
    cache->k_cache = malloc(cache_size * sizeof(float));
    cache->v_cache = malloc(cache_size * sizeof(float));
    if (!cache->k_cache || !cache->v_cache) {
        gemma4_kv_cache_free(cache);
        return false;
    }

    cache->seq_pos = (int *)calloc(1, sizeof(int));
    cache->n_seq = 1;
    return true;
}

void gemma4_kv_cache_free(gemma4_kv_cache_t *cache) {
    free(cache->k_cache);
    free(cache->v_cache);
    free(cache->seq_pos);
    memset(cache, 0, sizeof(*cache));
}

void gemma4_kv_cache_clear(gemma4_kv_cache_t *cache) {
    cache->seq_pos[0] = 0;
}

// Index into KV cache: k_cache[layer * max_ctx * n_kv_heads * head_dim + pos * n_kv_heads * head_dim + head * head_dim + dim]
static inline int64_t kv_offset(gemma4_kv_cache_t *c, int layer, int pos, int head, int dim) {
    return (int64_t)layer * c->max_ctx * c->n_kv_heads * c->head_dim
         + (int64_t)pos * c->n_kv_heads * c->head_dim
         + (int64_t)head * c->head_dim
         + dim;
}

// ========== Core Ops ==========

void gemma4_rmsnorm(float *out, const float *x, const float *weight, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * inv_rms * weight[i];
}

// RoPE: default (sliding) or proportional (full attention)
// If freq_factors != NULL, apply them (proportional rope)
// If partial_rot, only rotate first partial_rot_fraction of head_dim
void gemma4_rope(float *out, const float *x, int n_head, int head_dim, int pos,
                 float theta, float *freq_factors, bool partial_rot) {
    int rot_dim = partial_rot ? (int)(head_dim * GEMMA4_PARTIAL_ROT) : head_dim;
    // Ensure rot_dim is even
    rot_dim = rot_dim & ~1;

    for (int h = 0; h < n_head; h++) {
        const float *xh = x + h * head_dim;
        float *outh = out + h * head_dim;

        // Copy non-rotated dimensions
        for (int d = rot_dim; d < head_dim; d++)
            outh[d] = xh[d];

        // Apply RoPE to first rot_dim dimensions
        for (int d = 0; d < rot_dim; d += 2) {
            float freq;
            if (freq_factors) {
                // Proportional RoPE: freq = freqs[d/2] * pos / (theta^(d/head_dim))
                freq = freq_factors[d/2];
            } else {
                // Default RoPE: freq = pos / (theta^(d/head_dim))
                float inv_freq = powf(theta, (float)(-d) / (float)head_dim);
                freq = (float)pos * inv_freq;
            }
            float fcr = cosf(freq);
            float fci = sinf(freq);
            float v0 = xh[d];
            float v1 = xh[d+1];
            outh[d]   = v0 * fcr - v1 * fci;
            outh[d+1] = v0 * fci + v1 * fcr;
        }
    }
}

// GELU (tanh approximation)
void gemma4_gelu(float *out, const float *x, int n) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    for (int i = 0; i < n; i++) {
        float xv = x[i];
        out[i] = 0.5f * xv * (1.0f + tanhf(sqrt_2_over_pi * (xv + 0.044715f * xv * xv * xv)));
    }
}

// Final logit softcapping: tanh(x / cap) * cap
void gemma4_softcap(float *out, const float *x, int n, float cap) {
    float inv_cap = 1.0f / cap;
    for (int i = 0; i < n; i++)
        out[i] = tanhf(x[i] * inv_cap) * cap;
}

// ========== Matrix Multiply (simple F32) ==========

// C[M,N] = A[M,K] @ B[K,N]  (row-major)
static void gemm(float *c, const float *a, const float *b, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int t = 0; t < k; t++)
                sum += a[i * k + t] * b[t * n + j];
            c[i * n + j] = sum;
        }
    }
}

// C[N] = A[N,K] @ x[K]  (matrix-vector, row-major A)
static void gemv(float *c, const float *a, const float *x, int n, int k) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int t = 0; t < k; t++)
            sum += a[i * k + t] * x[t];
        c[i] = sum;
    }
}

// ========== Attention ==========

// Sliding window attention: only attend to last `sw` tokens
void gemma4_attention_sliding(float *out, const float *q, const float *k, const float *v,
                               int n_kv, int n_head, int n_kv_head, int head_dim, int sw) {
    int start_kv = n_kv > sw ? n_kv - sw : 0;
    int n_attended = n_kv - start_kv;

    for (int h = 0; h < n_head; h++) {
        int kv_h = h % n_kv_head;  // GQA: map Q head to KV head
        const float *qh = q + h * head_dim;
        float *outh = out + h * head_dim;

        // Compute attention scores
        float max_score = -1e30f;
        float scores[4096];  // max sliding window
        for (int t = start_kv; t < n_kv; t++) {
            const float *kt = k + kv_h * head_dim + (int64_t)t * n_kv_head * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; d++)
                s += qh[d] * kt[d];
            // Scale by 1/sqrt(head_dim) — Gemma 4 uses 1.0f, let's use standard
            s *= 1.0f / sqrtf((float)head_dim);
            if (s > max_score) max_score = s;
            scores[t - start_kv] = s;
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int t = 0; t < n_attended; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        float inv_sum = 1.0f / sum_exp;

        // Weighted sum of V
        memset(outh, 0, head_dim * sizeof(float));
        for (int t = start_kv; t < n_kv; t++) {
            const float *vt = v + kv_h * head_dim + (int64_t)t * n_kv_head * head_dim;
            float w = scores[t - start_kv] * inv_sum;
            for (int d = 0; d < head_dim; d++)
                outh[d] += w * vt[d];
        }
    }
}

// Full attention (no window)
void gemma4_attention_full(float *out, const float *q, const float *k, const float *v,
                            int n_kv, int n_head, int n_kv_head, int head_dim) {
    // Same as sliding but with no window limit
    gemma4_attention_sliding(out, q, k, v, n_kv, n_head, n_kv_head, head_dim, n_kv);
}

// ========== SwiGLU FFN ==========

static void gemma4_ffn(float *out, const float *x, const gemma4_ffn_quant_t *ffn, int d_model, int ffn_hidden) {
    // gate = x @ Wgate^T  [ffn_hidden]
    // up   = x @ Wup^T    [ffn_hidden]
    // out = (silu(gate) * up) @ Wdown^T  [d_model]
    float *gate = (float *)malloc(ffn_hidden * sizeof(float));
    float *up = (float *)malloc(ffn_hidden * sizeof(float));
    if (!gate || !up) { fprintf(stderr, "OOM in FFN\n"); free(gate); free(up); return; }

    gemv(gate, (const float *)ffn->gate, x, ffn_hidden, d_model);
    gemv(up, (const float *)ffn->up, x, ffn_hidden, d_model);

    // SiLU(x) = x * sigmoid(x)
    for (int i = 0; i < ffn_hidden; i++) {
        float sg = gate[i] / (1.0f + expf(-gate[i]));  // sigmoid
        up[i] = sg * up[i];
    }

    gemv(out, (const float *)ffn->down, up, d_model, ffn_hidden);

    free(gate);
    free(up);
}

// ========== Forward Pass (single token) ==========

void gemma4_forward(gemma4_model_t *model, gemma4_kv_cache_t *cache,
                    int token, int pos, float *logits) {
    const int n_embd = GEMMA4_D_MODEL;
    const int n_head = GEMMA4_N_HEADS;
    const int n_kv_head = GEMMA4_N_KV_HEADS;
    const int head_dim = GEMMA4_HEAD_DIM;
    const int n_ff = GEMMA4_FFN_HIDDEN;
    const int sliding_win = GEMMA4_SLIDING_WIN;

    // Token embedding
    float *hidden = (float *)malloc(n_embd * sizeof(float));
    float *cur = (float *)malloc(n_embd * sizeof(float));
    float *residual = (float *)malloc(n_embd * sizeof(float));
    float *q = (float *)malloc(n_head * head_dim * sizeof(float));
    float *k = (float *)malloc(n_kv_head * head_dim * sizeof(float));
    float *v = (float *)malloc(n_kv_head * head_dim * sizeof(float));
    float *q_rope = (float *)malloc(n_head * head_dim * sizeof(float));
    float *k_rope = (float *)malloc(n_kv_head * head_dim * sizeof(float));
    float *attn_out = (float *)malloc(n_head * head_dim * sizeof(float));
    float *ffn_in = (float *)malloc(n_embd * sizeof(float));
    float *ffn_out = (float *)malloc(n_embd * sizeof(float));

    if (!hidden || !cur || !residual || !q || !k || !v || !q_rope || !k_rope || !attn_out || !ffn_in || !ffn_out) {
        fprintf(stderr, "OOM in forward pass\n");
        free(hidden); free(cur); free(residual); free(q); free(k); free(v);
        free(q_rope); free(k_rope); free(attn_out); free(ffn_in); free(ffn_out);
        return;
    }

    // Embedding lookup
    const float *emb = model->token_embd_weight + (int64_t)token * n_embd;
    memcpy(hidden, emb, n_embd * sizeof(float));

    // Scale by sqrt(d_model)
    float scale = sqrtf((float)n_embd);
    for (int i = 0; i < n_embd; i++)
        hidden[i] *= scale;

    // Main transformer loop
    for (int il = 0; il < GEMMA4_N_LAYERS; il++) {
        gemma4_layer_t *layer = &model->layers[il];
        bool is_full = layer->layer_type == GEMMA4_LAYER_FULL;

        // Pre-attention RMSNorm
        gemma4_rmsnorm(cur, hidden, layer->attn_norm_weight, n_embd, GEMMA4_NORM_EPS);
        memcpy(residual, hidden, n_embd * sizeof(float));

        // Q projection
        gemv(q, layer->wq, cur, n_head * head_dim, n_embd);

        // Q norm (per-head RMSNorm)
        for (int h = 0; h < n_head; h++) {
            float *qh = q + h * head_dim;
            float *qnh = q_rope + h * head_dim;
            gemma4_rmsnorm(qnh, qh, layer->attn_q_norm_weight, head_dim, GEMMA4_NORM_EPS);
        }

        // RoPE on Q
        float theta = is_full ? GEMMA4_ROPE_THETA_FULL : GEMMA4_ROPE_THETA_SLIDE;
        gemma4_rope(q_rope, q_rope, n_head, head_dim, pos, theta,
                    layer->rope_freqs, is_full);

        // K/V projection
        gemv(k, layer->wk, cur, n_kv_head * head_dim, n_embd);

        // K norm
        for (int h = 0; h < n_kv_head; h++) {
            float *kh = k + h * head_dim;
            float *knh = k_rope + h * head_dim;
            gemma4_rmsnorm(knh, kh, layer->attn_k_norm_weight, head_dim, GEMMA4_NORM_EPS);
        }

        // RoPE on K
        gemma4_rope(k_rope, k_rope, n_kv_head, head_dim, pos, theta, layer->rope_freqs, is_full);

        // V = K if no separate V projection
        const float *v_cur;
        if (layer->wv == layer->wk) {
            // V norm directly on K (before RoPE)
            for (int h = 0; h < n_kv_head; h++) {
                float *vh = v + h * head_dim;
                gemma4_rmsnorm(vh, k + h * head_dim, layer->attn_k_norm_weight, head_dim, GEMMA4_NORM_EPS);
            }
            v_cur = v;
        } else {
            gemv(v, layer->wv, cur, n_kv_head * head_dim, n_embd);
            for (int h = 0; h < n_kv_head; h++) {
                float *vh = v + h * head_dim;
                gemma4_rmsnorm(vh, vh, layer->attn_k_norm_weight, head_dim, GEMMA4_NORM_EPS);
            }
            v_cur = v;
        }

        // Store K/V in cache
        int64_t off_k = kv_offset(cache, il, pos, 0, 0);
        memcpy((float *)cache->k_cache + off_k, k_rope, n_kv_head * head_dim * sizeof(float));
        int64_t off_v = kv_offset(cache, il, pos, 0, 0);
        memcpy((float *)cache->v_cache + off_v, v_cur, n_kv_head * head_dim * sizeof(float));
        cache->seq_pos[0] = pos + 1;

        // Attention
        int n_kv_total = cache->seq_pos[0];
        if (is_full) {
            gemma4_attention_full(attn_out, q_rope,
                (const float *)cache->k_cache + off_k - (int64_t)pos * n_kv_head * head_dim,
                (const float *)cache->v_cache + off_v - (int64_t)pos * n_kv_head * head_dim,
                n_kv_total, n_head, n_kv_head, head_dim);
        } else {
            gemma4_attention_sliding(attn_out, q_rope,
                (const float *)cache->k_cache + off_k - (int64_t)pos * n_kv_head * head_dim,
                (const float *)cache->v_cache + off_v - (int64_t)pos * n_kv_head * head_dim,
                n_kv_total, n_head, n_kv_head, head_dim, sliding_win);
        }

        // Output projection
        gemv(cur, layer->wo, attn_out, n_embd, n_head * head_dim);

        // Post-attention norm
        gemma4_rmsnorm(cur, cur, layer->post_attention_norm, n_embd, GEMMA4_NORM_EPS);

        // Residual
        for (int i = 0; i < n_embd; i++)
            cur[i] = cur[i] + residual[i];

        // FFN
        memcpy(ffn_in, cur, n_embd * sizeof(float));
        gemma4_ffn(ffn_out, ffn_in, &layer->ffn, n_embd, n_ff);

        // Post-FFN norm
        gemma4_rmsnorm(ffn_out, ffn_out, layer->post_ffw_norm_weight, n_embd, GEMMA4_NORM_EPS);

        // Residual + layer scale
        float ls = layer->out_scale;
        for (int i = 0; i < n_embd; i++)
            hidden[i] = ffn_out[i] * ls + cur[i];
    }

    // Final RMSNorm
    gemma4_rmsnorm(cur, hidden, model->output_norm_weight, n_embd, GEMMA4_NORM_EPS);

    // LM head
    gemv(logits, model->output_weight, cur, GEMMA4_VOCAB_SIZE, n_embd);

    // Final logit softcapping
    gemma4_softcap(logits, logits, GEMMA4_VOCAB_SIZE, GEMMA4_FINAL_SOFTCAP);

    free(hidden); free(cur); free(residual); free(q); free(k); free(v);
    free(q_rope); free(k_rope); free(attn_out); free(ffn_in); free(ffn_out);
}
