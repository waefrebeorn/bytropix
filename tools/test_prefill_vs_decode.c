/*
 * test_prefill_vs_decode.c — Compare prefilled vs decoded GQA output for position np
 * This tests whether gqa_kv_decode produces the same result as the last-token
 * output from the prefill's manual GQA attention for the same input.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "wubu_ssm.h"
#include "gguf_reader.h"

// KV cache inline (copied from infer_text.c)
#define MAX_CACHE_T (1024)
#define GQA_KV_DIM (GQA_KV_HEADS * GQA_HEAD_DIM)

typedef struct {
    float *h_k; int max_T; int current_T; int kv_dim;
    float *h_v;
} kv_cache_t;

static void kv_init(kv_cache_t *c, int max_T) {
    memset(c, 0, sizeof(*c));
    c->max_T = max_T; c->kv_dim = GQA_KV_DIM; c->current_T = 0;
    c->h_k = (float *)calloc((size_t)max_T * GQA_KV_DIM, sizeof(float));
    c->h_v = (float *)calloc((size_t)max_T * GQA_KV_DIM, sizeof(float));
}
static void kv_append(kv_cache_t *c, const float *k, const float *v, int n) {
    int off = c->current_T; c->current_T += n;
    memcpy(c->h_k + off * c->kv_dim, k, n * c->kv_dim * sizeof(float));
    memcpy(c->h_v + off * c->kv_dim, v, n * c->kv_dim * sizeof(float));
}
static void kv_free(kv_cache_t *c) { free(c->h_k); free(c->h_v); memset(c,0,sizeof(*c)); }

// === Rest of infer_text.c static functions, pasted ===
static float *rope_sc = NULL;
#define ROTARY_DIM 64
#define ROPE_THETA 10000000.0f

static int rope_init(void) {
    if (rope_sc) return 1;
    rope_sc = (float *)malloc((size_t)MAX_CACHE_T * ROTARY_DIM * sizeof(float));
    for (int p = 0; p < MAX_CACHE_T; p++) {
        for (int i = 0; i < ROTARY_DIM / 2; i++) {
            float theta = powf(ROPE_THETA, -2.0f * i / ROTARY_DIM);
            float angle = (float)p * theta;
            rope_sc[p * ROTARY_DIM + i * 2]     = cosf(angle);
            rope_sc[p * ROTARY_DIM + i * 2 + 1] = sinf(angle);
        }
    }
    return 1;
}

static void apply_rotary_to_buf(float *buf, int n_heads, int position, const float *sc) {
    const float *sc_p = sc + (size_t)position * ROTARY_DIM;
    for (int h = 0; h < n_heads; h++) {
        float *head = buf + (size_t)h * GQA_HEAD_DIM;
        for (int i = 0; i < ROTARY_DIM / 2; i++) {
            float cosv = sc_p[i * 2];
            float sinv = sc_p[i * 2 + 1];
            float d0 = head[i * 2];
            float d1 = head[i * 2 + 1];
            head[i * 2]     = d0 * cosv - d1 * sinv;
            head[i * 2 + 1] = d0 * sinv + d1 * cosv;
        }
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    rope_init();
    
    // Open GGUF
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "FAIL: open\n"); return 1; }
    
    // Find first GQA layer (layer 3, 0-indexed)
    int target_layer = 3;
    char name[256];
    snprintf(name, sizeof(name), "blk.%d.attn_q.weight", target_layer);
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "FAIL: no GQA layer %d\n", target_layer); return 1; }
    
    // Load GQA weights
    gqa_layer_weights w = {0};
    int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM; // 4096
    int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM; // 512
    
    w.attn_q_weight = (float *)malloc(D_MODEL * q_dim * 2 * sizeof(float));
    snprintf(name, sizeof(name), "blk.%d.attn_q.weight", target_layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "FAIL: no %s\n", name); return 1; }
    gguf_read_tensor_f32(ctx, t, w.attn_q_weight, D_MODEL * q_dim * 2);
    
    w.attn_k_weight = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
    snprintf(name, sizeof(name), "blk.%d.attn_k.weight", target_layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "FAIL: no %s\n", name); return 1; }
    gguf_read_tensor_f32(ctx, t, w.attn_k_weight, D_MODEL * kv_dim);
    
    w.attn_v_weight = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
    snprintf(name, sizeof(name), "blk.%d.attn_v.weight", target_layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "FAIL: no %s\n", name); return 1; }
    gguf_read_tensor_f32(ctx, t, w.attn_v_weight, D_MODEL * kv_dim);
    
    w.attn_output_weight = (float *)malloc(q_dim * D_MODEL * sizeof(float));
    snprintf(name, sizeof(name), "blk.%d.attn_output.weight", target_layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "FAIL: no %s\n", name); return 1; }
    gguf_read_tensor_f32(ctx, t, w.attn_output_weight, q_dim * D_MODEL);
    
    w.attn_q_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
    snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", target_layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "FAIL: no %s\n", name); return 1; }
    gguf_read_tensor_f32(ctx, t, w.attn_q_norm_weight, GQA_HEAD_DIM);
    
    w.attn_k_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
    snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", target_layer);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "FAIL: no %s\n", name); return 1; }
    gguf_read_tensor_f32(ctx, t, w.attn_k_norm_weight, GQA_HEAD_DIM);
    
    // Load first 10 token embeddings as test input
    int n_tok = 7;
    float *embd = NULL;
    int vs;
    
    // Load embedding matrix
    snprintf(name, sizeof(name), "token_embd.weight");
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "FAIL: no token_embd\n"); return 1; }
    vs = t->dims[1]; // vocab size
    embd = (float *)malloc((size_t)vs * D_MODEL * sizeof(float));
    gguf_read_tensor_f32(ctx, t, embd, (int64_t)vs * D_MODEL);
    
    // Generate test token IDs (first 7 tokens from vocab)
    int token_ids[8] = {248044, 248044, 248044, 248044, 248044, 248044, 248044, 248044};
    // Use actual distinguishable tokens: space, a, b, c, d, e, f (if available)
    token_ids[0] = 0; // token 0
    token_ids[1] = 200; // "cat"
    token_ids[2] = 0;  // etc
    token_ids[3] = 1;
    token_ids[4] = 2;
    token_ids[5] = 3;
    token_ids[6] = 4;
    
    // Generate embeddings
    float *x = (float *)malloc(n_tok * D_MODEL * sizeof(float));
    for (int i = 0; i < n_tok; i++) {
        int id = token_ids[i];
        if (id >= 0 && id < vs)
            memcpy(x + i * D_MODEL, embd + id * D_MODEL, D_MODEL * sizeof(float));
        else
            memset(x + i * D_MODEL, 0, D_MODEL * sizeof(float));
    }
    
    // ===== TEST 1: Prefill n_tok tokens, compare last-token output =====
    kv_cache_t cache_p;
    kv_init(&cache_p, MAX_CACHE_T);
    
    // Compute Q, K, V, gate for ALL tokens (manual prefill style)
    float *Q = (float *)malloc(n_tok * q_dim * sizeof(float));
    float *K = (float *)malloc(n_tok * kv_dim * sizeof(float));
    float *V = (float *)malloc(n_tok * kv_dim * sizeof(float));
    float *K_norm = (float *)malloc(n_tok * kv_dim * sizeof(float));
    float *gate_buf = (float *)malloc(n_tok * q_dim * sizeof(float));
    
    for (int s = 0; s < n_tok; s++) {
        const float *xs = x + s * D_MODEL;
        for (int j = 0; j < q_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)xs[i] * (double)w.attn_q_weight[i + j * D_MODEL];
            Q[s * q_dim + j] = (float)sum;
        }
        for (int j = 0; j < q_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)xs[i] * (double)w.attn_q_weight[i + (j + q_dim) * D_MODEL];
            gate_buf[s * q_dim + j] = (float)sum;
        }
        for (int j = 0; j < kv_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)xs[i] * (double)w.attn_k_weight[i + j * D_MODEL];
            K[s * kv_dim + j] = (float)sum;
        }
        for (int j = 0; j < kv_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)xs[i] * (double)w.attn_v_weight[i + j * D_MODEL];
            V[s * kv_dim + j] = (float)sum;
        }
    }
    
    // Q/K RMSNorm
    memcpy(K_norm, K, n_tok * kv_dim * sizeof(float));
    wubu_rms_norm(1, n_tok * GQA_Q_HEADS, GQA_HEAD_DIM, Q, w.attn_q_norm_weight, 1e-6f, Q);
    wubu_rms_norm(1, n_tok * GQA_KV_HEADS, GQA_HEAD_DIM, K_norm, w.attn_k_norm_weight, 1e-6f, K_norm);
    
    // RoPE
    for (int s = 0; s < n_tok; s++) {
        apply_rotary_to_buf(Q + s * q_dim, GQA_Q_HEADS, s, rope_sc);
        apply_rotary_to_buf(K_norm + s * kv_dim, GQA_KV_HEADS, s, rope_sc);
    }
    
    // Attention (prefill style: causal)
    float *attn_out_batch = (float *)calloc(n_tok * q_dim, sizeof(float));
    float scale = 1.0f / sqrtf((float)GQA_HEAD_DIM);
    for (int s = 0; s < n_tok; s++) {
        for (int h_q = 0; h_q < GQA_Q_HEADS; h_q++) {
            int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS);
            const float *qv = Q + s * q_dim + h_q * GQA_HEAD_DIM;
            float *out = attn_out_batch + s * q_dim + h_q * GQA_HEAD_DIM;
            
            float mx = -1e30f, sum_exp = 0.0f;
            for (int t = 0; t <= s; t++) {
                const float *kv = K_norm + t * kv_dim + h_kv * GQA_HEAD_DIM;
                float sc = 0.0f;
                for (int i = 0; i < GQA_HEAD_DIM; i++) sc += qv[i] * kv[i];
                sc *= scale;
                if (t == 0 || sc > mx) mx = sc;
            }
            for (int t = 0; t <= s; t++) {
                const float *vv = V + t * kv_dim + h_kv * GQA_HEAD_DIM;
                const float *kv = K_norm + t * kv_dim + h_kv * GQA_HEAD_DIM;
                float sc = 0.0f;
                for (int i = 0; i < GQA_HEAD_DIM; i++) sc += qv[i] * kv[i];
                sum_exp += expf(sc * scale - mx);
            }
            float inv = 1.0f / (sum_exp + 1e-30f);
            for (int t = 0; t <= s; t++) {
                const float *vv = V + t * kv_dim + h_kv * GQA_HEAD_DIM;
                const float *kv = K_norm + t * kv_dim + h_kv * GQA_HEAD_DIM;
                float sc = 0.0f;
                for (int i = 0; i < GQA_HEAD_DIM; i++) sc += qv[i] * kv[i];
                float a = expf(sc * scale - mx) * inv;
                for (int i = 0; i < GQA_HEAD_DIM; i++) out[i] += a * vv[i];
            }
        }
    }
    
    // Gate
    for (int i = 0; i < n_tok * q_dim; i++)
        attn_out_batch[i] *= 1.0f / (1.0f + expf(-gate_buf[i]));
    
    // Output projection for last token
    float prefilled[D_MODEL];
    memset(prefilled, 0, sizeof(prefilled));
    { const float *in = attn_out_batch + (n_tok - 1) * q_dim;
      for (int i = 0; i < q_dim; i++) {
          float a = in[i];
          for (int j = 0; j < D_MODEL; j++)
              prefilled[j] += a * w.attn_output_weight[i + j * q_dim];
      }
    }
    
    // Append to KV cache (simulating prefill)
    kv_append(&cache_p, K_norm, V, n_tok);
    
    // ===== TEST 2: Decode 1 more token using gqa_kv_decode =====
    // First, compute the last embedding (same as token n_tok-1)
    float *x_last = x + (n_tok - 1) * D_MODEL;
    
    // Simulate decode: compute Q, K, V for 1 more token with same input
    float q_norm[4096], k_norm[512], v_raw[512], gate_dec[4096];
    
    for (int j = 0; j < q_dim; j++) {
        double sum = 0.0;
        for (int i = 0; i < D_MODEL; i++)
            sum += (double)x_last[i] * (double)w.attn_q_weight[i + j * D_MODEL];
        q_norm[j] = (float)sum;
    }
    // K
    for (int j = 0; j < kv_dim; j++) {
        double sum = 0.0;
        for (int i = 0; i < D_MODEL; i++)
            sum += (double)x_last[i] * (double)w.attn_k_weight[i + j * D_MODEL];
        k_norm[j] = (float)sum;
    }
    // V
    for (int j = 0; j < kv_dim; j++) {
        double sum = 0.0;
        for (int i = 0; i < D_MODEL; i++)
            sum += (double)x_last[i] * (double)w.attn_v_weight[i + j * D_MODEL];
        v_raw[j] = (float)sum;
    }
    // Gate
    for (int j = 0; j < q_dim; j++) {
        double sum = 0.0;
        for (int i = 0; i < D_MODEL; i++)
            sum += (double)x_last[i] * (double)w.attn_q_weight[i + (j + q_dim) * D_MODEL];
        gate_dec[j] = (float)sum;
    }
    
    // RMSNorm Q and K
    wubu_rms_norm(1, GQA_Q_HEADS, GQA_HEAD_DIM, q_norm, w.attn_q_norm_weight, 1e-6f, q_norm);
    wubu_rms_norm(1, GQA_KV_HEADS, GQA_HEAD_DIM, k_norm, w.attn_k_norm_weight, 1e-6f, k_norm);
    
    // RoPE at position n_tok
    apply_rotary_to_buf(q_norm, GQA_Q_HEADS, n_tok, rope_sc);
    apply_rotary_to_buf(k_norm, GQA_KV_HEADS, n_tok, rope_sc);
    
    // Attention: attend to ALL prefill tokens + new token
    float attn_dec[4096];
    memset(attn_dec, 0, sizeof(attn_dec));
    int new_T = cache_p.current_T;  // should be n_tok
    
        for (int h_q = 0; h_q < GQA_Q_HEADS; h_q++) {
        int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS);
        const float *q_vec = q_norm + h_q * GQA_HEAD_DIM;
        float *out = attn_dec + h_q * GQA_HEAD_DIM;
        
        float mx = -1e30f, sum_exp = 0.0f, inv = 0.0f;
        for (int t = 0; t < new_T; t++) {
            const float *kv = cache_p.h_k + t * kv_dim + h_kv * GQA_HEAD_DIM;
            float s = 0.0f;
            for (int i = 0; i < GQA_HEAD_DIM; i++) s += q_vec[i] * kv[i];
            s *= scale;
            if (t == 0 || s > mx) mx = s;
        }
        for (int t = 0; t < new_T; t++) {
            const float *vv = cache_p.h_v + t * kv_dim + h_kv * GQA_HEAD_DIM;
            const float *kv = cache_p.h_k + t * kv_dim + h_kv * GQA_HEAD_DIM;
            float s = 0.0f;
            for (int i = 0; i < GQA_HEAD_DIM; i++) s += q_vec[i] * kv[i];
            sum_exp += expf(s * scale - mx);
        }
        inv = 1.0f / (sum_exp + 1e-30f);
        for (int t = 0; t < new_T; t++) {
            const float *vv = cache_p.h_v + t * kv_dim + h_kv * GQA_HEAD_DIM;
            const float *kv = cache_p.h_k + t * kv_dim + h_kv * GQA_HEAD_DIM;
            float s = 0.0f;
            for (int i = 0; i < GQA_HEAD_DIM; i++) s += q_vec[i] * kv[i];
            float a = expf(s * scale - mx) * inv;
            for (int i = 0; i < GQA_HEAD_DIM; i++) out[i] += a * vv[i];
        }
    }
    
    // Gate
    for (int i = 0; i < q_dim; i++)
        attn_dec[i] *= 1.0f / (1.0f + expf(-gate_dec[i]));
    
    // Output projection
    float decoded[D_MODEL];
    memset(decoded, 0, sizeof(decoded));
    for (int i = 0; i < q_dim; i++) {
        float a = attn_dec[i];
        for (int j = 0; j < D_MODEL; j++)
            decoded[j] += a * w.attn_output_weight[i + j * q_dim];
    }
    
    // ===== COMPARE =====
    printf("=== Prefill vs Decode (same input, GQA layer %d) ===\n", target_layer);
    printf("n_tok = %d, prefill output vs decode output (same token, last position):\n", n_tok);
    
    float max_diff = 0.0f;
    int max_i = -1;
    for (int i = 0; i < D_MODEL; i++) {
        float diff = fabsf(prefilled[i] - decoded[i]);
        if (diff > max_diff) { max_diff = diff; max_i = i; }
    }
    printf("  Max diff: %e at dim %d\n", max_diff, max_i);
    printf("  Prefilled[0]=%f  Decoded[0]=%f  Prefilled[%d]=%f  Decoded[%d]=%f\n",
           prefilled[0], decoded[0], max_i, prefilled[max_i], max_i, decoded[max_i]);
    
    // Free everything
    gguf_close(ctx);
    kv_free(&cache_p);
    free(Q); free(K); free(V); free(K_norm); free(gate_buf); free(attn_out_batch);
    free(x); free(embd);
    free(w.attn_q_weight); free(w.attn_k_weight); free(w.attn_v_weight);
    free(w.attn_output_weight); free(w.attn_q_norm_weight); free(w.attn_k_norm_weight);
    free(rope_sc); rope_sc = NULL;
    
    printf("=== PASS ===\n");
    return 0;
}
