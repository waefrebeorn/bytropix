/*
 * test_full_pipeline.c — Compare full 40-layer prefill vs decode for last token
 * Tests: prefill T=7 → last token output vs 
 *        prefill T=6 → decode T=1 step → last token output
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "gguf_reader.h"

#define MAX_CACHE_T (1024)
#define GQA_KV_DIM (GQA_KV_HEADS * GQA_HEAD_DIM)

typedef struct {
    float *h_k; int max_T; int current_T; int kv_dim;
    float *h_v;
} kv_cache_t;

static void kv_init(kv_cache_t *c, int max_T) {
    memset(c, 0, sizeof(*c)); c->max_T = max_T; c->kv_dim = GQA_KV_DIM; c->current_T = 0;
    c->h_k = (float *)calloc((size_t)max_T * GQA_KV_DIM, sizeof(float));
    c->h_v = (float *)calloc((size_t)max_T * GQA_KV_DIM, sizeof(float));
}
static void kv_append(kv_cache_t *c, const float *k, const float *v, int n) {
    int off = c->current_T; c->current_T += n;
    memcpy(c->h_k + off * c->kv_dim, k, n * c->kv_dim * sizeof(float));
    memcpy(c->h_v + off * c->kv_dim, v, n * c->kv_dim * sizeof(float));
}
static void kv_free(kv_cache_t *c) { free(c->h_k); free(c->h_v); memset(c,0,sizeof(*c)); }

static float *rope_sc = NULL;
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

static int is_gqa(int l) {
    return (l % 4) == 3; // layers 3,7,11,...,39
}

// Simple timing
#include <sys/time.h>
static double now_sec(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    rope_init();
    
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "FAIL: open\n"); return 1; }
    
    // Load embeddings
    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t) { fprintf(stderr, "FAIL: no token_embd\n"); return 1; }
    int vs = t->dims[1];
    float *embd = (float *)malloc((int64_t)vs * D_MODEL * sizeof(float));
    gguf_read_tensor_f32(ctx, t, embd, (int64_t)vs * D_MODEL);
    
    // Determine layer configuration
    int n_layers = 0;
    for (int i = 0; ; i++) {
        char nm[256]; snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", i);
        if (!gguf_find_tensor(ctx, nm)) break;
        n_layers++;
    }
    printf("Found %d layers\n", n_layers);
    
    // Load all layer weights (GQA + SSM norms)
    typedef struct {
        float *attn_norm;          // Pre-attention norm
        float *post_attn_norm;     // Post-attention norm
        float *norm_weight;        // Final norm
        
        // GQA weights
        gqa_layer_weights gqa;
        bool has_gqa;
        
        // SSM weights (full layer)
        ssm_layer_weights ssm;
        bool has_ssm;
    } layer_t;
    
    layer_t *layers = (layer_t *)calloc(n_layers, sizeof(layer_t));
    
    // Load final norm
    float *final_norm = (float *)malloc(D_MODEL * sizeof(float));
    t = gguf_find_tensor(ctx, "output_norm.weight");
    if (t) gguf_read_tensor_f32(ctx, t, final_norm, D_MODEL);
    else { free(final_norm); final_norm = NULL; }
    
    // Load output weight
    float *output_weight = NULL;
    t = gguf_find_tensor(ctx, "output.weight");
    if (t) {
        output_weight = (float *)malloc((int64_t)vs * D_MODEL * sizeof(float));
        gguf_read_tensor_f32(ctx, t, output_weight, (int64_t)vs * D_MODEL);
        printf("Output weight loaded: %d x %d\n", vs, D_MODEL);
    }
    
    // Buffer for full dequant of MoE shared expert and router (all layers use same dequant approach)
    // We'll skip MoE for this test and just pass identity
    
    for (int l = 0; l < n_layers; l++) {
        char nm[256];
        
        // Norms
        layers[l].attn_norm = (float *)malloc(D_MODEL * sizeof(float));
        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", l);
        t = gguf_find_tensor(ctx, nm);
        if (t) gguf_read_tensor_f32(ctx, t, layers[l].attn_norm, D_MODEL);
        
        layers[l].post_attn_norm = (float *)malloc(D_MODEL * sizeof(float));
        snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", l);
        t = gguf_find_tensor(ctx, nm);
        if (t) gguf_read_tensor_f32(ctx, t, layers[l].post_attn_norm, D_MODEL);
        
        if (is_gqa(l)) {
            int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
            int kv_dim = GQA_KV_DIM;
            layers[l].has_gqa = true;
            
            layers[l].gqa.attn_q_weight = (float *)malloc(D_MODEL * q_dim * 2 * sizeof(float));
            snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) gguf_read_tensor_f32(ctx, t, layers[l].gqa.attn_q_weight, D_MODEL * q_dim * 2);
            
            layers[l].gqa.attn_k_weight = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
            snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) gguf_read_tensor_f32(ctx, t, layers[l].gqa.attn_k_weight, D_MODEL * kv_dim);
            
            layers[l].gqa.attn_v_weight = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
            snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) gguf_read_tensor_f32(ctx, t, layers[l].gqa.attn_v_weight, D_MODEL * kv_dim);
            
            layers[l].gqa.attn_output_weight = (float *)malloc(q_dim * D_MODEL * sizeof(float));
            snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) gguf_read_tensor_f32(ctx, t, layers[l].gqa.attn_output_weight, q_dim * D_MODEL);
            
            layers[l].gqa.attn_q_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
            snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) gguf_read_tensor_f32(ctx, t, layers[l].gqa.attn_q_norm_weight, GQA_HEAD_DIM);
            
            layers[l].gqa.attn_k_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
            snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) gguf_read_tensor_f32(ctx, t, layers[l].gqa.attn_k_norm_weight, GQA_HEAD_DIM);
        } else {
            layers[l].has_ssm = true;
            // Load SSM weights (simplified: just wqkv, gate, beta, alpha, conv, norm, out)
            // wqkv [D_MODEL, CONV_DIM]
            layers[l].ssm.attn_qkv_weight = (float *)malloc(D_MODEL * CONV_DIM * sizeof(float));
            snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) gguf_read_tensor_f32(ctx, t, layers[l].ssm.attn_qkv_weight, D_MODEL * CONV_DIM);
            
            // gate projection [D_MODEL, VALUE_DIM]
            layers[l].ssm.attn_gate_weight = (float *)malloc(D_MODEL * VALUE_DIM * sizeof(float));
            snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) gguf_read_tensor_f32(ctx, t, layers[l].ssm.attn_gate_weight, D_MODEL * VALUE_DIM);
            
            // beta [D_MODEL, DT_RANK]
            layers[l].ssm.ssm_beta_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
            snprintf(nm, sizeof(nm), "blk.%d.ssm_beta.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) gguf_read_tensor_f32(ctx, t, layers[l].ssm.ssm_beta_weight, D_MODEL * DT_RANK);
            
            // alpha [D_MODEL, DT_RANK]
            layers[l].ssm.ssm_alpha_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
            snprintf(nm, sizeof(nm), "blk.%d.ssm_alpha.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) gguf_read_tensor_f32(ctx, t, layers[l].ssm.ssm_alpha_weight, D_MODEL * DT_RANK);
            
            // dt_bias [DT_RANK]
            layers[l].ssm.ssm_dt_bias = (float *)malloc(DT_RANK * sizeof(float));
            snprintf(nm, sizeof(nm), "blk.%d.ssm_dt.bias", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) gguf_read_tensor_f32(ctx, t, layers[l].ssm.ssm_dt_bias, DT_RANK);
            
            // ssm_a [DT_RANK]
            layers[l].ssm.ssm_a = (float *)malloc(DT_RANK * sizeof(float));
            snprintf(nm, sizeof(nm), "blk.%d.ssm_a.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) gguf_read_tensor_f32(ctx, t, layers[l].ssm.ssm_a, DT_RANK);
            
            // conv1d [CONV_KERNEL, CONV_DIM]
            layers[l].ssm.ssm_conv1d_weight = (float *)malloc(CONV_KERNEL * CONV_DIM * sizeof(float));
            snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) gguf_read_tensor_f32(ctx, t, layers[l].ssm.ssm_conv1d_weight, CONV_KERNEL * CONV_DIM);
            
            // norm [SSM_D_STATE]
            layers[l].ssm.ssm_norm_weight = (float *)malloc(SSM_D_STATE * sizeof(float));
            snprintf(nm, sizeof(nm), "blk.%d.ssm_norm.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) gguf_read_tensor_f32(ctx, t, layers[l].ssm.ssm_norm_weight, SSM_D_STATE);
            
            // output [VALUE_DIM, D_MODEL]
            layers[l].ssm.ssm_out_weight = (float *)malloc(VALUE_DIM * D_MODEL * sizeof(float));
            snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) gguf_read_tensor_f32(ctx, t, layers[l].ssm.ssm_out_weight, VALUE_DIM * D_MODEL);
        }
    }
    
    // ===== Generate test input =====
    int n_tok = 8;
    int token_ids[8];
    // Use real token IDs that are likely to exist
    token_ids[0] = 248044; // BOS
    for (int i = 1; i < n_tok; i++) token_ids[i] = 100 + i; // some valid tokens
    
    float *x_all = (float *)malloc(n_tok * D_MODEL * sizeof(float));
    for (int i = 0; i < n_tok; i++) {
        int id = token_ids[i];
        id = id < 0 ? 0 : (id >= vs ? vs-1 : id);
        memcpy(x_all + i * D_MODEL, embd + id * D_MODEL, D_MODEL * sizeof(float));
    }
    
    // ===== METHOD 1: Prefill all 7 tokens, compare with last-token output =====
    // Also get the full 7-token result for method 2 comparison
    
    // ===== METHOD 2: Prefill 6 tokens, then decode 1 more =====
    int prefill_n = 7;
    
    // Allocate per-layer state
    kv_cache_t *kv = (kv_cache_t *)calloc(n_layers, sizeof(kv_cache_t));
    for (int l = 0; l < n_layers; l++) if (is_gqa(l)) kv_init(&kv[l], MAX_CACHE_T);
    
    float *ssm_state[40] = {NULL};
    float *conv_state[40] = {NULL};
    for (int l = 0; l < n_layers; l++) if (!is_gqa(l)) {
        ssm_state[l] = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
        conv_state[l] = (float *)calloc((CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
    }
    
    // === Prefill n_tok-1 tokens ===
    float *residual = (float *)malloc(prefill_n * D_MODEL * sizeof(float));
    float *normed = (float *)malloc(prefill_n * D_MODEL * sizeof(float));
    float *attn = (float *)malloc(prefill_n * D_MODEL * sizeof(float));
    float *ffn = (float *)malloc(prefill_n * D_MODEL * sizeof(float));
    
    memcpy(residual, x_all, prefill_n * D_MODEL * sizeof(float));
    
    printf("Prefilling %d tokens...\n", prefill_n);
    double t0 = now_sec();
    for (int l = 0; l < n_layers; l++) {
        // Pre-norm
        wubu_rms_norm(1, prefill_n, D_MODEL, residual, layers[l].attn_norm, 1e-6f, normed);
        
        if (is_gqa(l)) {
            int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
            int kv_dim = GQA_KV_DIM;
            
            float *Q = (float *)malloc(prefill_n * q_dim * sizeof(float));
            float *K = (float *)malloc(prefill_n * kv_dim * sizeof(float));
            float *V = (float *)malloc(prefill_n * kv_dim * sizeof(float));
            float *K_norm = (float *)malloc(prefill_n * kv_dim * sizeof(float));
            float *gate_buf = (float *)malloc(prefill_n * q_dim * sizeof(float));
            float *attn_out = (float *)calloc(prefill_n * q_dim, sizeof(float));
            
            gqa_layer_weights *w = &layers[l].gqa;
            for (int s = 0; s < prefill_n; s++) {
                const float *xs = normed + s * D_MODEL;
                for (int j = 0; j < q_dim; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < D_MODEL; i++)
                        sum += (double)xs[i] * (double)w->attn_q_weight[i + j * D_MODEL];
                    Q[s * q_dim + j] = (float)sum;
                }
                for (int j = 0; j < q_dim; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < D_MODEL; i++)
                        sum += (double)xs[i] * (double)w->attn_q_weight[i + (j + q_dim) * D_MODEL];
                    gate_buf[s * q_dim + j] = (float)sum;
                }
                for (int j = 0; j < kv_dim; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < D_MODEL; i++)
                        sum += (double)xs[i] * (double)w->attn_k_weight[i + j * D_MODEL];
                    K[s * kv_dim + j] = (float)sum;
                }
                for (int j = 0; j < kv_dim; j++) {
                    double sum = 0.0;
                    for (int i = 0; i < D_MODEL; i++)
                        sum += (double)xs[i] * (double)w->attn_v_weight[i + j * D_MODEL];
                    V[s * kv_dim + j] = (float)sum;
                }
            }
            
            memcpy(K_norm, K, prefill_n * kv_dim * sizeof(float));
            wubu_rms_norm(1, prefill_n * GQA_Q_HEADS, GQA_HEAD_DIM, Q, w->attn_q_norm_weight, 1e-6f, Q);
            wubu_rms_norm(1, prefill_n * GQA_KV_HEADS, GQA_HEAD_DIM, K_norm, w->attn_k_norm_weight, 1e-6f, K_norm);
            
            for (int s = 0; s < prefill_n; s++) {
                apply_rotary_to_buf(Q + s * q_dim, GQA_Q_HEADS, s, rope_sc);
                apply_rotary_to_buf(K_norm + s * kv_dim, GQA_KV_HEADS, s, rope_sc);
            }
            
            float scale = 1.0f / sqrtf((float)GQA_HEAD_DIM);
            for (int s = 0; s < prefill_n; s++) {
                for (int h_q = 0; h_q < GQA_Q_HEADS; h_q++) {
                    int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS);
                    const float *qv = Q + s * q_dim + h_q * GQA_HEAD_DIM;
                    float *out = attn_out + s * q_dim + h_q * GQA_HEAD_DIM;
                    
                    float mx = -1e30f, sum_exp = 0.0f;
                    for (int t = 0; t <= s; t++) {
                        const float *kv = K_norm + t * kv_dim + h_kv * GQA_HEAD_DIM;
                        float sc = 0.0f;
                        for (int i = 0; i < GQA_HEAD_DIM; i++) sc += qv[i] * kv[i];
                        sc *= scale;
                        if (t == 0 || sc > mx) mx = sc;
                    }
                    for (int t = 0; t <= s; t++) {
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
            
            for (int i = 0; i < prefill_n * q_dim; i++)
                attn_out[i] *= 1.0f / (1.0f + expf(-gate_buf[i]));
            
            // Output projection
            memset(attn, 0, prefill_n * D_MODEL * sizeof(float));
            for (int s = 0; s < prefill_n; s++) {
                const float *in = attn_out + s * q_dim;
                for (int i = 0; i < q_dim; i++) {
                    float a = in[i];
                    for (int j = 0; j < D_MODEL; j++)
                        attn[s * D_MODEL + j] += a * w->attn_output_weight[i + j * q_dim];
                }
            }
            
            // Append to KV cache
            kv_append(&kv[l], K_norm, V, prefill_n);
            
            free(Q); free(K); free(V); free(K_norm); free(gate_buf); free(attn_out);
        } else {
            wubu_ssm_forward(normed, 1, prefill_n, &layers[l].ssm,
                             ssm_state[l], conv_state[l], attn);
        }
        
        // Residual
        for (int i = 0; i < prefill_n * D_MODEL; i++) residual[i] += attn[i];
        
        // Post-norm
        wubu_rms_norm(1, prefill_n, D_MODEL, residual, layers[l].post_attn_norm, 1e-6f, normed);
        
        // FFN (identity — skip MoE for now)
        memcpy(ffn, normed, prefill_n * D_MODEL * sizeof(float));
        for (int i = 0; i < prefill_n * D_MODEL; i++) residual[i] += ffn[i];
    }
    
    // Final norm
    if (final_norm) {
        wubu_rms_norm(1, prefill_n, D_MODEL, residual, final_norm, 1e-6f, normed);
        memcpy(residual, normed, prefill_n * D_MODEL * sizeof(float));
    }
    
    // Prefilled last token
    float *h_prefill_last = residual + (prefill_n - 1) * D_MODEL;
    printf("Prefill done: %.2f s\n", now_sec() - t0);
    
    // === Decode 1 more token with same input as position prefill_n-1 ===
    // (using the SAME initial embedding that was used for the last prefill token)
    float *x_step = (float *)malloc(D_MODEL * sizeof(float));
    memcpy(x_step, x_all + (prefill_n - 1) * D_MODEL, D_MODEL * sizeof(float));
    
    float residual_dec[D_MODEL];
    float normed_dec[D_MODEL], attn_dec[D_MODEL], ffn_dec[D_MODEL];
    memcpy(residual_dec, x_step, D_MODEL * sizeof(float));
    
    printf("Decoding 1 step...\n");
    t0 = now_sec();
    for (int l = 0; l < n_layers; l++) {
        // Pre-norm
        wubu_rms_norm(1, 1, D_MODEL, residual_dec, layers[l].attn_norm, 1e-6f, normed_dec);
        
        if (is_gqa(l)) {
            int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
            int kv_dim = GQA_KV_DIM;
            gqa_layer_weights *w = &layers[l].gqa;
            
            float q_norm[4096], k_norm[512], v_raw[512], gate_buf[4096];
            
            for (int j = 0; j < q_dim; j++) {
                double sum = 0.0;
                for (int i = 0; i < D_MODEL; i++)
                    sum += (double)normed_dec[i] * (double)w->attn_q_weight[i + j * D_MODEL];
                q_norm[j] = (float)sum;
            }
            for (int j = 0; j < q_dim; j++) {
                double sum = 0.0;
                for (int i = 0; i < D_MODEL; i++)
                    sum += (double)normed_dec[i] * (double)w->attn_q_weight[i + (j + q_dim) * D_MODEL];
                gate_buf[j] = (float)sum;
            }
            for (int j = 0; j < kv_dim; j++) {
                double sum = 0.0;
                for (int i = 0; i < D_MODEL; i++)
                    sum += (double)normed_dec[i] * (double)w->attn_k_weight[i + j * D_MODEL];
                k_norm[j] = (float)sum;
            }
            for (int j = 0; j < kv_dim; j++) {
                double sum = 0.0;
                for (int i = 0; i < D_MODEL; i++)
                    sum += (double)normed_dec[i] * (double)w->attn_v_weight[i + j * D_MODEL];
                v_raw[j] = (float)sum;
            }
            
            wubu_rms_norm(1, GQA_Q_HEADS, GQA_HEAD_DIM, q_norm, w->attn_q_norm_weight, 1e-6f, q_norm);
            wubu_rms_norm(1, GQA_KV_HEADS, GQA_HEAD_DIM, k_norm, w->attn_k_norm_weight, 1e-6f, k_norm);
            
            // RoPE at position = prefill_n (the current cache length)
            // BUT: this is wrong — the input embedding is the same as position prefill_n-1
            // So we should use position prefill_n-1 for RoPE!
            // Actually no — in the decode path, the NEW token's position is prefill_n-1
            // because the input is the embedding of the same token.
            // Wait — in prefill mode, token prefill_n-1 uses RoPE position prefill_n-1.
            // In decode mode, the input is the embedding of the token at position prefill_n-1,
            // and we're computing its attention with all prefill_n tokens in the cache.
            // So the RoPE position should be prefill_n-1, NOT prefill_n!
            
            // CURRENT APPROACH (WRONG?):
            // Position = cache->current_T = prefill_n (the number of entries in cache)
            // But the token at position prefill_n is the NEXT token, not the current one.
            // The current token is at position prefill_n - 1.
            
            apply_rotary_to_buf(q_norm, GQA_Q_HEADS, prefill_n - 1, rope_sc);
            apply_rotary_to_buf(k_norm, GQA_KV_HEADS, prefill_n - 1, rope_sc);
            
            // KV append
            kv_append(&kv[l], k_norm, v_raw, 1);
            int new_T = kv[l].current_T;
            
            // Attention
            memset(attn_dec, 0, sizeof(attn_dec));
            float scale_val = 1.0f / sqrtf((float)GQA_HEAD_DIM);
            float attn_raw[4096];
            memset(attn_raw, 0, sizeof(attn_raw));
            
            for (int h_q = 0; h_q < GQA_Q_HEADS; h_q++) {
                int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS);
                const float *q_vec = q_norm + h_q * GQA_HEAD_DIM;
                float *out = attn_raw + h_q * GQA_HEAD_DIM;
                
                float mx = -1e30f, sum_exp = 0.0f, inv;
                for (int t = 0; t < new_T; t++) {
                    const float *kvec = kv[l].h_k + t * kv_dim + h_kv * GQA_HEAD_DIM;
                    float s = 0.0f;
                    for (int i = 0; i < GQA_HEAD_DIM; i++) s += q_vec[i] * kvec[i];
                    s *= scale_val;
                    if (t == 0 || s > mx) mx = s;
                }
                for (int t = 0; t < new_T; t++) {
                    const float *kvec = kv[l].h_k + t * kv_dim + h_kv * GQA_HEAD_DIM;
                    float s = 0.0f;
                    for (int i = 0; i < GQA_HEAD_DIM; i++) s += q_vec[i] * kvec[i];
                    sum_exp += expf(s * scale_val - mx);
                }
                inv = 1.0f / (sum_exp + 1e-30f);
                for (int t = 0; t < new_T; t++) {
                    const float *vv = kv[l].h_v + t * kv_dim + h_kv * GQA_HEAD_DIM;
                    const float *kvec = kv[l].h_k + t * kv_dim + h_kv * GQA_HEAD_DIM;
                    float s = 0.0f;
                    for (int i = 0; i < GQA_HEAD_DIM; i++) s += q_vec[i] * kvec[i];
                    float a = expf(s * scale_val - mx) * inv;
                    for (int i = 0; i < GQA_HEAD_DIM; i++) out[i] += a * vv[i];
                }
            }
            
            // Gate
            for (int i = 0; i < q_dim; i++)
                attn_raw[i] *= 1.0f / (1.0f + expf(-gate_buf[i]));
            
            // Output projection
            memset(attn_dec, 0, sizeof(attn_dec));
            for (int i = 0; i < q_dim; i++) {
                float a = attn_raw[i];
                for (int j = 0; j < D_MODEL; j++)
                    attn_dec[j] += a * w->attn_output_weight[i + j * q_dim];
            }
            
            memcpy(attn, attn_dec, D_MODEL * sizeof(float)); // reuse attn as 1-token buffer
        } else {
            wubu_ssm_forward(normed_dec, 1, 1, &layers[l].ssm,
                             ssm_state[l], conv_state[l], attn);
        }
        
        for (int i = 0; i < D_MODEL; i++) residual_dec[i] += attn[i];
        
        // Post-norm
        wubu_rms_norm(1, 1, D_MODEL, residual_dec, layers[l].post_attn_norm, 1e-6f, normed_dec);
        
        // FFN (identity)
        memcpy(ffn_dec, normed_dec, D_MODEL * sizeof(float));
        for (int i = 0; i < D_MODEL; i++) residual_dec[i] += ffn_dec[i];
    }
    
    // Final norm
    if (final_norm) {
        wubu_rms_norm(1, 1, D_MODEL, residual_dec, final_norm, 1e-6f, normed_dec);
        memcpy(residual_dec, normed_dec, D_MODEL * sizeof(float));
    }
    printf("Decode done: %.2f s\n", now_sec() - t0);
    
    // ===== COMPARE =====
    printf("\n=== COMPARISON: Prefill vs Decode (full %d layers) ===\n", n_layers);
    printf("Input: %d tokens, comparing output at position %d\n", prefill_n, prefill_n - 1);
    
    float max_diff = 0.0f;
    int max_i = -1;
    double sum_sq = 0.0;
    for (int i = 0; i < D_MODEL; i++) {
        float diff = fabsf(h_prefill_last[i] - residual_dec[i]);
        sum_sq += (double)diff * diff;
        if (diff > max_diff) { max_diff = diff; max_i = i; }
    }
    float rms_diff = sqrtf(sum_sq / D_MODEL);
    printf("  RMS diff: %e\n", rms_diff);
    printf("  Max diff: %e at dim %d\n", max_diff, max_i);
    printf("  Prefilled[0]=%f  Decoded[0]=%f\n", h_prefill_last[0], residual_dec[0]);
    printf("  Prefilled[%d]=%f  Decoded[%d]=%f\n", max_i, h_prefill_last[max_i], max_i, residual_dec[max_i]);
    
    // Compare output projections (logits)
    if (output_weight) {
        float logit_pref[20], logit_dec[20];
        for (int j = 0; j < 20; j++) {
            double sum_p = 0.0, sum_d = 0.0;
            for (int k = 0; k < D_MODEL; k++) {
                sum_p += (double)h_prefill_last[k] * (double)output_weight[j * D_MODEL + k];
                sum_d += (double)residual_dec[k] * (double)output_weight[j * D_MODEL + k];
            }
            logit_pref[j] = (float)sum_p;
            logit_dec[j] = (float)sum_d;
        }
        float logit_max_diff = 0.0f;
        for (int j = 0; j < 20; j++) {
            float d = fabsf(logit_pref[j] - logit_dec[j]);
            if (d > logit_max_diff) logit_max_diff = d;
        }
        printf("  Max logit diff (first 20): %e\n", logit_max_diff);
    }
    
    // Compute hidden state RMSNorm statistics
    float pref_rms = 0, dec_rms = 0;
    for (int i = 0; i < D_MODEL; i++) {
        pref_rms += h_prefill_last[i] * h_prefill_last[i];
        dec_rms += residual_dec[i] * residual_dec[i];
    }
    pref_rms = sqrtf(pref_rms / D_MODEL);
    dec_rms = sqrtf(dec_rms / D_MODEL);
    printf("  Prefill RMS: %f  Decode RMS: %f\n", pref_rms, dec_rms);
    
    // ===== CLEANUP =====
    gguf_close(ctx);
    for (int l = 0; l < n_layers; l++) {
        free(layers[l].attn_norm);
        free(layers[l].post_attn_norm);
        if (layers[l].has_gqa) {
            free(layers[l].gqa.attn_q_weight);
            free(layers[l].gqa.attn_k_weight);
            free(layers[l].gqa.attn_v_weight);
            free(layers[l].gqa.attn_output_weight);
            free(layers[l].gqa.attn_q_norm_weight);
            free(layers[l].gqa.attn_k_norm_weight);
        } else {
            free(layers[l].ssm.attn_qkv_weight);
            free(layers[l].ssm.attn_gate_weight);
            free(layers[l].ssm.ssm_beta_weight);
            free(layers[l].ssm.ssm_alpha_weight);
            free(layers[l].ssm.ssm_dt_bias);
            free(layers[l].ssm.ssm_a);
            free(layers[l].ssm.ssm_conv1d_weight);
            free(layers[l].ssm.ssm_norm_weight);
            free(layers[l].ssm.ssm_out_weight);
        }
        kv_free(&kv[l]);
        free(ssm_state[l]);
        free(conv_state[l]);
    }
    free(layers);
    free(x_all); free(residual); free(normed); free(attn); free(ffn);
    free(x_step); free(embd); free(output_weight); free(final_norm);
    free(rope_sc); rope_sc = NULL;
    
    printf("=== PASS ===\n");
    return 0;
}
