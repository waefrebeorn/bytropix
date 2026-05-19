/**
 * infer_text.c — Full text generation pipeline v2
 *
 * Phase 1 (prefill): full forward over prompt tokens, populate KV caches.
 * Phase 2 (decode): token-by-token using GQA KV cache + SSM state carry.
 * Lazy per-expert MoE cache (dequant router + shared expert once).
 *
 * Usage: ./infer_text [gguf_path] ["prompt"] [max_tokens] [top_k]
 * Env:  MOE=1  VERBOSE=1  NOGPU=1
 */
#include "wubu_model.h"
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include "wubu_tokenizer.h"
#include "gguf_reader.h"
#include "bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <stdbool.h>
#include <omp.h>

// ================================================================
// GQA KV Cache
// ================================================================
#define MAX_CACHE_T (262144)
#define GQA_KV_DIM (GQA_KV_HEADS * GQA_HEAD_DIM) // 512

// Debug dump env vars
#define DUMP_DIR_VAR "DUMP_LAYER_DIR"
#define DUMP_HIDDEN_VAR "DUMP_HIDDEN_NORM"
static void maybe_dump_layer(int l, float *residual, int np) {
    const char *dir = getenv(DUMP_DIR_VAR);
    if (dir) {
        char fn[512];
        snprintf(fn, sizeof(fn), "%s/layer_%02d.bin", dir, l);
        FILE *f = fopen(fn, "wb");
        if (f) { fwrite(residual, sizeof(float), np * D_MODEL, f); fclose(f); }
    }
}
static void maybe_dump_hidden(float *residual, int np) {
    const char *path = getenv(DUMP_HIDDEN_VAR);
    if (path) {
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(residual + (np - 1) * D_MODEL, sizeof(float), D_MODEL, f); fclose(f); }
    }
}

typedef struct {
    float *h_k;        // [current_T, kv_dim] post-RMSNorm K
    float *h_v;        // [current_T, kv_dim] raw V
    int max_T;
    int current_T;
    int kv_dim;
} kv_cache_t;

static int kv_init(kv_cache_t *c, int max_T, int kv_dim) {
    memset(c, 0, sizeof(*c));
    c->max_T = max_T;
    c->kv_dim = kv_dim;
    c->current_T = 0;
    size_t bytes = (size_t)max_T * kv_dim * sizeof(float);
    c->h_k = (float *)malloc(bytes);
    c->h_v = (float *)malloc(bytes);
    if (!c->h_k || !c->h_v) return 0;
    return 1;
}

static void kv_append(kv_cache_t *c, const float *k, const float *v, int n) {
    int off = c->current_T;
    int nt = off + n;
    size_t nb = (size_t)n * c->kv_dim * sizeof(float);
    memcpy(c->h_k + off * c->kv_dim, k, nb);
    memcpy(c->h_v + off * c->kv_dim, v, nb);
    c->current_T = nt;
}

static void kv_free(kv_cache_t *c) {
    free(c->h_k); free(c->h_v);
    memset(c, 0, sizeof(*c));
}

// ================================================================
// RoPE sin/cos table (pre-computed for up to MAX_CACHE_T positions)
// ================================================================
static float *rope_sc = NULL;

// MRoPE (Multi-Resolution RoPE) for Qwen3.6-35B-A3B.
// Sections: [11, 11, 10] pairs. Each section restarts frequency.
// In each section: theta[i] = base^(-2*i / section_dim) for i=0..n_pairs-1
// where section_dim = 2 * n_pairs (22, 22, 20).
// Rotation uses split-half pairing: dim[pair*2] with dim[pair*2+1]
// Ref: llama.cpp ggml_mrope_cache_init for GGML_ROPE_TYPE_MROPE

static int rope_init(void) {
    if (rope_sc) return 1;
    rope_sc = (float *)malloc((size_t)MAX_CACHE_T * ROTARY_DIM * sizeof(float));
    if (!rope_sc) return 0;
    
    // MRoPE sections: [11, 11, 10] pairs
    static const int mrope_sections[MRoPE_SECTIONS] = {MRoPE_SEC0_PAIRS, MRoPE_SEC1_PAIRS, MRoPE_SEC2_PAIRS};

    for (int p = 0; p < MAX_CACHE_T; p++) {
        int pair_idx = 0;
        for (int s = 0; s < MRoPE_SECTIONS; s++) {
            int n_pairs = mrope_sections[s];
            int sec_dim = 2 * n_pairs;  // section dimension (22, 22, or 20)
            for (int i = 0; i < n_pairs; i++) {
                float theta = powf(ROPE_THETA, -2.0f * i / sec_dim);
                float angle = (float)p * theta;
                rope_sc[p * ROTARY_DIM + pair_idx * 2]     = cosf(angle);
                rope_sc[p * ROTARY_DIM + pair_idx * 2 + 1] = sinf(angle);
                pair_idx++;
            }
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

// ================================================================
// GQA decode: Q for 1 token, attend vs cached K/V
// ================================================================
static void gqa_kv_decode(
    const float *x_step, const gqa_layer_weights *w,
    kv_cache_t *cache, float *output)
{
    int kv_dim = GQA_KV_DIM;
    int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
    float q_raw[4096], q_norm[4096], k_raw[512], k_norm[512], v_raw[512];

    // Q proj
        #pragma omp parallel for
        for (int j = 0; j < q_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_step[i] * (double)w->attn_q_weight[i + j * D_MODEL];
            q_raw[j] = (float)sum;
    }
    memcpy(q_norm, q_raw, q_dim * sizeof(float));
    wubu_rms_norm(1, GQA_Q_HEADS, GQA_HEAD_DIM, q_norm, w->attn_q_norm_weight, 1e-6f, q_norm);

    // K proj
        #pragma omp parallel for
        for (int j = 0; j < kv_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_step[i] * (double)w->attn_k_weight[i + j * D_MODEL];
        k_raw[j] = (float)sum;
    }
    memcpy(k_norm, k_raw, kv_dim * sizeof(float));
    wubu_rms_norm(1, GQA_KV_HEADS, GQA_HEAD_DIM, k_norm, w->attn_k_norm_weight, 1e-6f, k_norm);

    // Apply RoPE to Q and K (in-place) at position cache->current_T
    apply_rotary_to_buf(q_norm, GQA_Q_HEADS, cache->current_T, rope_sc);
    apply_rotary_to_buf(k_norm, GQA_KV_HEADS, cache->current_T, rope_sc);

    // V proj
        #pragma omp parallel for
        for (int j = 0; j < kv_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_step[i] * (double)w->attn_v_weight[i + j * D_MODEL];
        v_raw[j] = (float)sum;
    }

    // Append K,V to cache
    kv_append(cache, k_norm, v_raw, 1);
    int new_T = cache->current_T;

    // Gate proj
    float gate[4096];
        #pragma omp parallel for
        for (int j = 0; j < q_dim; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += (double)x_step[i] * (double)w->attn_q_weight[i + (j + q_dim) * D_MODEL];
        gate[j] = (float)sum;
    }

    // Attention
    float scale = 1.0f / sqrtf((float)GQA_HEAD_DIM);
    float *attn_out = (float *)calloc(q_dim, sizeof(float));
    for (int h_q = 0; h_q < GQA_Q_HEADS; h_q++) {
        int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS);
        const float *q_vec = q_norm + h_q * GQA_HEAD_DIM;
        float *out = attn_out + h_q * GQA_HEAD_DIM;

        float mx = -1e30f, sum_exp = 0.0f;
        for (int t = 0; t < new_T; t++) {
            const float *kv = cache->h_k + t * kv_dim + h_kv * GQA_HEAD_DIM;
            float s = 0.0f;
            for (int i = 0; i < GQA_HEAD_DIM; i++) s += q_vec[i] * kv[i];
            s *= scale;
            if (t == 0 || s > mx) mx = s;
        }
        for (int t = 0; t < new_T; t++) {
            const float *kv = cache->h_k + t * kv_dim + h_kv * GQA_HEAD_DIM;
            float s = 0.0f;
            for (int i = 0; i < GQA_HEAD_DIM; i++) s += q_vec[i] * kv[i];
            s = expf(s * scale - mx);
            sum_exp += s;
        }
        float inv = 1.0f / (sum_exp + 1e-30f);
        for (int t = 0; t < new_T; t++) {
            const float *vv = cache->h_v + t * kv_dim + h_kv * GQA_HEAD_DIM;
            const float *kv = cache->h_k + t * kv_dim + h_kv * GQA_HEAD_DIM;
            float s = 0.0f;
            for (int i = 0; i < GQA_HEAD_DIM; i++) s += q_vec[i] * kv[i];
            float a = expf(s * scale - mx) * inv;
            for (int i = 0; i < GQA_HEAD_DIM; i++) out[i] += a * vv[i];
        }
    }

    // Gate
    for (int i = 0; i < q_dim; i++)
        attn_out[i] *= 1.0f / (1.0f + expf(-gate[i]));

    // Output projection
    memset(output, 0, D_MODEL * sizeof(float));
    #pragma omp parallel for
    for (int j = 0; j < D_MODEL; j++) {
        double sum = 0.0;
        for (int i = 0; i < q_dim; i++)
            sum += (double)attn_out[i] * (double)w->attn_output_weight[i + j * q_dim];
        output[j] = (float)sum;
    }
    free(attn_out);
}

// ================================================================
// SSM decode: 1 token with state carry
// ================================================================
static void ssm_kv_decode(
    const float *x_step, const ssm_layer_weights *w,
    float *ssm_state, float *conv_state, float *output)
{
    // wubu_ssm_forward with T=1, carrying state in/out
    wubu_ssm_forward(x_step, 1, 1, w, ssm_state, conv_state, output, NULL, NULL);
}

// ================================================================
// Lazy MoE cache
// ================================================================
typedef struct {
    int eid;
    float *gate, *up, *down;
} lexpert_t;

typedef struct {
    lexpert_t *exps;
    int n, cap;
    float *sh_gate, *sh_up, *sh_down, *router, *sh_gate_proj;
    const uint8_t *q_gate, *q_up, *q_down;
    int64_t raw_sz, raw_sz_d;
    int ty_ge, ty_gd, ty_gi, ty_gs;
    bool has;
} lmoe_t;

static void lmoe_init(lmoe_t *m) { memset(m, 0, sizeof(*m)); }
static void lmoe_free(lmoe_t *m) {
    for (int i = 0; i < m->n; i++) { free(m->exps[i].gate); free(m->exps[i].up); free(m->exps[i].down); }
    free(m->exps); free(m->sh_gate); free(m->sh_up); free(m->sh_down); free(m->router); free(m->sh_gate_proj);
    memset(m, 0, sizeof(*m));
}

// ================================================================
// Lazy MoE forward: cache experts across steps, no 3GB temp arrays
// ================================================================
// Uses moe_expert_forward directly from cached per-expert arrays.
// No contiguous 256-expert arrays needed — looks up experts by ID.

// Dequant ONE expert from interleaved [D_MODEL, D_FF, N_EXPERTS] tensor.
// q_full: raw quantized data for ALL experts (interleaved, N_EXPERTS=256)
// Contiguous-per-expert dequant: each block of 256 elements stores consecutive
// values WITHIN one expert, NOT across experts.
// The full tensor for N_EXPERTS experts stores each expert's data contiguously:
//   expert 0: blocks 0..(ne/256)-1
//   expert 1: blocks (ne/256)..2*(ne/256)-1
//   ...
// Expert eid's data starts at q_full + eid * gguf_raw_size(type, ne).
static void dequant_one_expert_contiguous(const uint8_t *q_full, int type, int eid, int64_t ne, float *out) {
    int block_bytes = (int)gguf_raw_size(type, 256);
    int64_t raw_per_exp = gguf_raw_size(type, ne);
    if (block_bytes <= 0 || raw_per_exp <= 0) {
        fprintf(stderr, "dequant_one_expert_contiguous: unsupported type %d\n", type);
        memset(out, 0, ne * sizeof(float));
        return;
    }
    const uint8_t *edata = q_full + eid * raw_per_exp;
    for (int64_t b = 0; b < ne; b += 256) {
        int64_t nr = ne - b;
        int64_t nb = (nr > 256) ? 256 : nr;
        gguf_dequantize(edata + (b/256) * block_bytes, type, nb, out + b);
    }
}

static void dequant_multi_expert_contiguous(const uint8_t *q_full, int type, int64_t ne,
                                              const int *eids, int n_eids, float **outputs) {
    int block_bytes = (int)gguf_raw_size(type, 256);
    int64_t raw_per_exp = gguf_raw_size(type, ne);
    if (block_bytes <= 0 || raw_per_exp <= 0) {
        fprintf(stderr, "dequant_multi_expert_contiguous: unsupported type %d\n", type);
        for (int i = 0; i < n_eids; i++) memset(outputs[i], 0, ne * sizeof(float));
        return;
    }
    #pragma omp parallel for
    for (int i = 0; i < n_eids; i++) {
        int eid = eids[i];
        const uint8_t *edata = q_full + eid * raw_per_exp;
        for (int64_t b = 0; b < ne; b += 256) {
            int64_t nr = ne - b;
            int64_t nb = (nr > 256) ? 256 : nr;
            gguf_dequantize(edata + (b/256) * block_bytes, type, nb, outputs[i] + b);
        }
    }
}

static void moe_expert_forward_lazy(const float *x, const float *gate_w,
                                     const float *up_w, const float *down_w,
                                     float *temp, float *output) {
    // temp: [D_FF * 3] scratch
    float *gate_out = temp;
    float *up_out   = temp + D_FF;
    float *act_out  = temp + D_FF * 2;

    // gate = x @ gate_w  [2048] @ [2048, 512] -> [512]
    #pragma omp parallel for
    for (int j = 0; j < D_FF; j++) {
        float sum = 0.0f;
        for (int k = 0; k < D_MODEL; k++)
            sum += x[k] * gate_w[k + j * D_MODEL];
        gate_out[j] = sum;
    }

    // up = x @ up_w
    #pragma omp parallel for
    for (int j = 0; j < D_FF; j++) {
        float sum = 0.0f;
        for (int k = 0; k < D_MODEL; k++)
            sum += x[k] * up_w[k + j * D_MODEL];
        up_out[j] = sum;
    }

    // act = silu(gate) * up
    #pragma omp parallel for
    for (int j = 0; j < D_FF; j++) {
        float g = gate_out[j];
        float silu = (g < -80.0f) ? 0.0f : g / (1.0f + expf(-g));
        act_out[j] = silu * up_out[j];
    }

    // out = act @ down_w  [512] @ [512, 2048] -> [2048]
    #pragma omp parallel for
    for (int j = 0; j < D_MODEL; j++) {
        float sum = 0.0f;
        for (int k = 0; k < D_FF; k++)
            sum += act_out[k] * down_w[k + j * D_FF];
        output[j] = sum;
    }
}

// Find cached expert by ID (linear search — small n, typically 8)
static float* find_cached(lmoe_t *mc, int eid, int which) {
    // which: 0=gate, 1=up, 2=down
    for (int i = 0; i < mc->n; i++)
        if (mc->exps[i].eid == eid)
            return which == 0 ? mc->exps[i].gate :
                   which == 1 ? mc->exps[i].up : mc->exps[i].down;
    return NULL;
}

static void lazy_moe_decode(
    const float *x, int B, int T,
    const uint8_t *q_gate_exps, const uint8_t *q_up_exps, const uint8_t *q_down_exps,
    lmoe_t *mc, float *output)
{
    int N = B * T;
    float *scores = (float *)malloc(N * N_EXPERTS * sizeof(float));
    wubu_moe_router(x, B, T, mc->router, scores);

    int topk_indices[N * N_ACTIVE_EXPTS];
    float topk_weights[N * N_ACTIVE_EXPTS];

    for (int s = 0; s < N; s++) {
        float *sc = scores + s * N_EXPERTS;
        float mx = sc[0];
        for (int e = 1; e < N_EXPERTS; e++) if (sc[e] > mx) mx = sc[e];

        float sum_exp = 0.0f;
        for (int e = 0; e < N_EXPERTS; e++) sum_exp += expf(sc[e] - mx);
        float inv = 1.0f / (sum_exp + 1e-30f);

        float sm[256];
        for (int e = 0; e < N_EXPERTS; e++) sm[e] = expf(sc[e] - mx) * inv;

        int *is = topk_indices + s * N_ACTIVE_EXPTS;
        float *ws = topk_weights + s * N_ACTIVE_EXPTS;

        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            int best_i = -1; float best_v = -1e30f;
            for (int e = 0; e < N_EXPERTS; e++) {
                int used = 0;
                for (int pk = 0; pk < k; pk++) if (is[pk] == e) { used = 1; break; }
                if (!used && sm[e] > best_v) { best_v = sm[e]; best_i = e; }
            }
            is[k] = best_i; ws[k] = best_v;
        }
        float sw = 0; for (int k = 0; k < N_ACTIVE_EXPTS; k++) sw += ws[k];
        if (sw > 1e-30f) { float iw = 1.0f / sw; for (int k = 0; k < N_ACTIVE_EXPTS; k++) ws[k] *= iw; }
    }

    // Collect unique expert IDs
    int unique_ids[N_ACTIVE_EXPTS * N];
    int n_unique = 0;
    for (int s = 0; s < N; s++) {
        int *is = topk_indices + s * N_ACTIVE_EXPTS;
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            int eid = is[k]; if (eid < 0) continue;
            int seen = 0;
            for (int u = 0; u < n_unique; u++) if (unique_ids[u] == eid) { seen = 1; break; }
            if (!seen) unique_ids[n_unique++] = eid;
        }
    }

    // Check if routing changed — need to dequant new experts?
    int changed = (n_unique != mc->n);
    if (!changed) {
        for (int u = 0; u < n_unique; u++)
            if (unique_ids[u] != mc->exps[u].eid) { changed = 1; break; }
    }

    // Dequant only if routing changed
    if (changed) {
        for (int i = 0; i < mc->n; i++) { free(mc->exps[i].gate); free(mc->exps[i].up); free(mc->exps[i].down); }
        mc->n = 0;
        if (!mc->exps || mc->cap < n_unique) {
            free(mc->exps);
            mc->exps = (lexpert_t *)malloc(n_unique * sizeof(lexpert_t));
            mc->cap = n_unique;
        }
        // Use optimized multi-expert block-dequant: dequant each block once,
        // extract all active experts' elements from each block.
        int64_t ne = (int64_t)D_MODEL * D_FF;
        int64_t nd = (int64_t)D_FF * D_MODEL;
        
        // Allocate per-expert buffers
        for (int u = 0; u < n_unique; u++) {
            int eid = unique_ids[u];
            mc->exps[u].eid = eid;
            mc->exps[u].gate = (float *)malloc(ne * sizeof(float));
            mc->exps[u].up = (float *)malloc(ne * sizeof(float));
            mc->exps[u].down = (float *)malloc(nd * sizeof(float));
        }
        
        // Dequant gate_exps, up_exps, down_exps using interleaved block approach
        // Build arrays of output pointers and eids for batch dequant
        float **gate_outs = (float **)malloc(n_unique * sizeof(float *));
        float **up_outs = (float **)malloc(n_unique * sizeof(float *));
        float **down_outs = (float **)malloc(n_unique * sizeof(float *));
        for (int u = 0; u < n_unique; u++) {
            gate_outs[u] = mc->exps[u].gate;
            up_outs[u] = mc->exps[u].up;
            down_outs[u] = mc->exps[u].down;
        }
        
        // Fix: down_exps may use a different quant type (e.g., L39 is IQ4_XS)
        // Use ty_ge for gate/up, ty_gd (or ty_ge fallback) for down
        int down_type = mc->ty_gd ? mc->ty_gd : mc->ty_ge;
        
        dequant_multi_expert_contiguous(q_gate_exps, mc->ty_ge, ne, unique_ids, n_unique, gate_outs);
        dequant_multi_expert_contiguous(q_up_exps, mc->ty_ge, ne, unique_ids, n_unique, up_outs);
        dequant_multi_expert_contiguous(q_down_exps, down_type, nd, unique_ids, n_unique, down_outs);
        
        free(gate_outs);
        free(up_outs);
        free(down_outs);
        mc->n = n_unique;
    }

    // Direct forward: no 3GB temp arrays, lookup per-expert from cache
    // Scratch for moe_expert_forward_lazy per-token
    float *scratch = (float *)malloc(D_FF * 3 * sizeof(float));

    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_MODEL;
        float *out_s = output + s * D_MODEL;
        int *is = topk_indices + s * N_ACTIVE_EXPTS;
        float *ws = topk_weights + s * N_ACTIVE_EXPTS;

        // Shared expert
        if (mc->sh_gate && mc->sh_up && mc->sh_down) {
            float *sg = scratch;
            float *su = scratch + D_FF;
            float *sa = scratch + D_FF * 2;
            #pragma omp parallel for
            for (int j = 0; j < SHARED_D_FF; j++) {
                float sum = 0.0f;
                for (int k = 0; k < D_MODEL; k++)
                    sum += x_s[k] * mc->sh_gate[k + j * D_MODEL];
                sg[j] = sum;
            }
            #pragma omp parallel for
            for (int j = 0; j < SHARED_D_FF; j++) {
                float sum = 0.0f;
                for (int k = 0; k < D_MODEL; k++)
                    sum += x_s[k] * mc->sh_up[k + j * D_MODEL];
                su[j] = sum;
            }
            #pragma omp parallel for
            for (int j = 0; j < SHARED_D_FF; j++) {
                float g = sg[j];
                sa[j] = ((g < -80.0f) ? 0.0f : g / (1.0f + expf(-g))) * su[j];
            }
            #pragma omp parallel for
            for (int j = 0; j < D_MODEL; j++) {
                float sum = 0.0f;
                for (int k = 0; k < SHARED_D_FF; k++)
                    sum += sa[k] * mc->sh_down[k + j * SHARED_D_FF];
                out_s[j] = sum;
            }
            // Apply shared expert gate: sigmoid(x @ ffn_gate_inp_shexp.weight)
            if (mc->sh_gate_proj) {
                float gate_sum = 0.0f;
                for (int k = 0; k < D_MODEL; k++)
                    gate_sum += x_s[k] * mc->sh_gate_proj[k];
                float gate_val = 1.0f / (1.0f + expf(-gate_sum));
                for (int j = 0; j < D_MODEL; j++)
                    out_s[j] *= gate_val;
            }
        } else {
            memset(out_s, 0, D_MODEL * sizeof(float));
        }

        // Routed experts
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            int e = is[k];
            float wgt = ws[k];
            if (e < 0 || wgt < 1e-30f) continue;

            float *gate_w = find_cached(mc, e, 0);
            float *up_w   = find_cached(mc, e, 1);
            float *down_w = find_cached(mc, e, 2);
            if (!gate_w || !up_w || !down_w) continue;

            float expert_out[2048];
            moe_expert_forward_lazy(x_s, gate_w, up_w, down_w, scratch, expert_out);

            for (int j = 0; j < D_MODEL; j++)
                out_s[j] += wgt * expert_out[j];
        }
    }

    free(scratch);
    free(scores);
}

// ================================================================
// Model loaders
// ================================================================
static float *load_embd(gguf_ctx *ctx, int *vocab_sz) {
    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t) return NULL;
    int64_t ne = 1;
    for (int i = 0; i < t->n_dims; i++) ne *= t->dims[i];
    *vocab_sz = (int)(ne / D_MODEL);
    float *e = (float *)malloc(ne * sizeof(float));
    if (!e || gguf_read_tensor_f32(ctx, t, e, ne) <= 0) { free(e); return NULL; }
    return e;
}

static int sample_greedy(const float *logits, int vs) {
    int b = 0; float bv = logits[0];
    for (int i = 1; i < vs; i++) if (logits[i] > bv) { bv = logits[i]; b = i; }
    return b;
}

static int sample(float *logits, int vs, float temperature, int top_k, float top_p,
                  const int *recent_tokens, int n_recent, float rep_penalty) {
    // Greedy fallback for temperature <= 0
    if (temperature <= 0.0f) {
        int best = 0; float bv = logits[0];
        for (int i = 1; i < vs; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }

    // Repetition penalty: scale down logits of recent tokens
    if (rep_penalty > 1.0f && n_recent > 0) {
        for (int r = 0; r < n_recent; r++) {
            int id = recent_tokens[r];
            if (id >= 0 && id < vs) {
                if (logits[id] > 0.0f) logits[id] /= rep_penalty;
                else logits[id] *= rep_penalty;
            }
        }
    }

    // Working copy with temperature scaling + subtract max for stability
    float *probs = (float *)malloc(vs * sizeof(float));
    float max_l = logits[0];
    for (int i = 1; i < vs; i++) if (logits[i] > max_l) max_l = logits[i];
    for (int i = 0; i < vs; i++) probs[i] = (logits[i] - max_l) / temperature;

    // Top-K: keep only the top K logits (zero out the rest by setting to -inf)
    if (top_k > 0 && top_k < vs) {
        float *vals = (float *)malloc(vs * sizeof(float));
        memcpy(vals, probs, vs * sizeof(float));
        for (int i = 0; i < top_k; i++) {
            int best = i;
            for (int j = i+1; j < vs; j++) if (vals[j] > vals[best]) best = j;
            float t = vals[i]; vals[i] = vals[best]; vals[best] = t;
        }
        float threshold = vals[top_k - 1];
        free(vals);
        for (int i = 0; i < vs; i++)
            if (probs[i] < threshold) probs[i] = -1e30f;
    }

    // Softmax
    max_l = probs[0];
    for (int i = 1; i < vs; i++) if (probs[i] > max_l) max_l = probs[i];
    float sum = 0.0f;
    for (int i = 0; i < vs; i++) { probs[i] = expf(probs[i] - max_l); sum += probs[i]; }
    float inv_sum = 1.0f / (sum + 1e-30f);
    for (int i = 0; i < vs; i++) probs[i] *= inv_sum;

    // Top-P (nucleus): keep smallest set whose cumulative prob >= top_p
    if (top_p > 0.0f && top_p < 1.0f) {
        // Only sort entries that survived top-K — max K entries, not full vocab
        int nz = 0;
        for (int i = 0; i < vs; i++) if (probs[i] > 0.0f) nz++;
        if (nz > 0) {
            typedef struct { float p; int i; } pair_t;
            pair_t *pairs = (pair_t *)malloc(nz * sizeof(pair_t));
            int idx = 0;
            for (int i = 0; i < vs; i++)
                if (probs[i] > 0.0f) { pairs[idx].p = probs[i]; pairs[idx].i = i; idx++; }
            // Simple sort (nz is at most top_k=20, so O(nz²) is negligible)
            for (int i = 0; i < nz; i++) {
                int best = i;
                for (int j = i+1; j < nz; j++) if (pairs[j].p > pairs[best].p) best = j;
                pair_t t = pairs[i]; pairs[i] = pairs[best]; pairs[best] = t;
            }
            float cum = 0.0f;
            int cutoff = nz;
            for (int i = 0; i < nz; i++) {
                if (pairs[i].p <= 0.0f) break;
                cum += pairs[i].p;
                if (cum >= top_p) { cutoff = i + 1; break; }
            }
            // Zero out excluded entries in original probs
            for (int i = cutoff; i < nz; i++) probs[pairs[i].i] = 0.0f;
            free(pairs);
            // Renormalize
            sum = 0.0f;
            for (int i = 0; i < vs; i++) if (probs[i] > 0.0f) sum += probs[i];
            if (sum > 1e-30f) { inv_sum = 1.0f / sum; for (int i = 0; i < vs; i++) if (probs[i] > 0.0f) probs[i] *= inv_sum; }
        }
    }

    // Sample from the resulting distribution
    float r = (float)rand() / RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < vs; i++) {
        cum += probs[i];
        if (r <= cum) { free(probs); return i; }
    }
    free(probs);
    return vs - 1; // fallback (should not reach here)
}

static volatile int stop = 0;
static void cleanup_gpu(void) {
    // Force cleanup all CUDA resources — resets the device, freeing all allocations
    cudaDeviceReset();
}
static void handler(int s) { (void)s; stop = 1; }

// ================================================================
// Main
// ================================================================
typedef struct {
    bool is_gqa;
    union {
        gqa_layer_weights gqa;
        ssm_layer_weights ssm;
    } w;
    bool moe;
    lmoe_t lm;
    kv_cache_t kv;
    float *ssm_state, *conv_state; // per-layer SSM state (carried between steps)
    // GPU pointers (valid if use_gpu)
    gpu_ssm_weights gpu_ssm_w;
    gpu_gqa_weights gpu_gqa_w;
    float *d_ssm_state, *d_conv_state; // GPU SSM state
    float *d_scratch_decode;           // GPU scratch for decode (1 token)
    float *d_Q_full, *d_K, *d_V, *d_attn_out; // GPU GQA decode scratch
} layer_ctx_t;

int main(int argc, char **argv) {
    const char *path = (argc > 1 && strlen(argv[1])) ? argv[1]
        : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *prompt = argc > 2 ? argv[2] : "The meaning of life is";
    int max_tok = argc > 3 ? atoi(argv[3]) : 64;
    int top_k = argc > 4 ? atoi(argv[4]) : 1;
    int verb = getenv("VERBOSE") ? atoi(getenv("VERBOSE")) : 0;
    int moe_on = getenv("MOE") ? atoi(getenv("MOE")) : 1;
    int moe_max_l = getenv("MOE_LAYERS") ? atoi(getenv("MOE_LAYERS")) : 0;
    int no_gpu = getenv("NOGPU") ? 1 : 0;
    float temperature = getenv("TEMP") ? atof(getenv("TEMP")) : 1.0f;
    int samp_top_k = getenv("TOP_K") ? atoi(getenv("TOP_K")) : 20;
    float top_p = getenv("TOP_P") ? atof(getenv("TOP_P")) : 0.95f;
    float rep_penalty = getenv("REP_PENALTY") ? atof(getenv("REP_PENALTY")) : 1.0f;
    int chat_mode = getenv("CHAT") ? atoi(getenv("CHAT")) : 0;

    signal(SIGINT, handler);
    srand(time(NULL));

    // Pre-compute RoPE sin/cos table
    if (!rope_init()) { fprintf(stderr, "Failed to allocate RoPE table\n"); return 1; }

    // Disable stdout buffering for real-time debug
    setvbuf(stdout, NULL, _IONBF, 0);
    
    printf("=== infer_text v2 — KV Cache + SSM Carry + Lazy MoE ===\n");
    printf("Model: %s\nPrompt: \"%s\" | max=%d | topk=%d | MOE=%d%s\n",
           path, prompt, max_tok, top_k, moe_on,
           chat_mode ? " | CHAT=1" : "");
    printf("Sampling: temp=%.1f top_k=%d top_p=%.2f\n", temperature, samp_top_k, top_p);

    double T0 = now_sec();

    // ---- GGUF ----
    double t0 = now_sec();
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    printf("GGUF: %.2f s\n", now_sec() - t0);

    // ---- Tokenizer ----
    t0 = now_sec();
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, path)) return 1;
    printf("Tok: %d voc, %d merg (%.2f s)\n", tok.vocab_size, tok.n_merges, now_sec() - t0);

    // ---- Model ----
    t0 = now_sec();
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, path)) return 1;
    if (mdl.gguf_ctx) gguf_close(mdl.gguf_ctx);
    mdl.gguf_ctx = ctx;
    printf("Model: %.2f s | %d layers | %s / %s\n",
           now_sec() - t0, mdl.n_layers,
           mdl.layers[0].is_ssm ? "SSM" : "GQA",
           mdl.layers[mdl.n_layers-1].is_ssm ? "SSM" : "GQA");

    // ---- Embeddings ----
    t0 = now_sec();
    int vs;
    float *embd = load_embd(ctx, &vs);
    if (!embd) { fprintf(stderr, "No embeddings\n"); return 1; }
    printf("Embd: %d tok (%.2f s)\n", vs, now_sec() - t0);

    // ---- GPU init ----
    cublasHandle_t ch = NULL;
    cudaStream_t st = NULL;
    float *d_out_w = NULL, *d_hid = NULL, *d_log = NULL;
    float *d_sincos = NULL;
    int use_gpu = 0;
    if (mdl.output_weight && !no_gpu) {
        // Check GPU memory
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        printf("GPU VRAM: %.0f MB free / %.0f MB total\n",
               free_mem / (1024.0*1024.0), total_mem / (1024.0*1024.0));
        cublasCreate(&ch); cudaStreamCreate(&st);
        d_out_w = gpu_upload_output_weight(ch, mdl.output_weight, vs, st);
        cudaMalloc(&d_hid, D_MODEL * sizeof(float));
        cudaMalloc(&d_log, vs * sizeof(float));
        use_gpu = 1;
        printf("GPU out: on\n");
        // Pre-compute RoPE for GPU (up to max_cache positions)
        cudaMalloc(&d_sincos, MAX_CACHE_T * ROTARY_DIM * sizeof(float));
        if (d_sincos) {
            wubu_cuda_precompute_rotary(MAX_CACHE_T, d_sincos, st);
            cudaStreamSynchronize(st);
        }
    } else {
        printf("GPU out: off\n");
    }

    // ---- Build per-layer contexts ----
    layer_ctx_t *lc = (layer_ctx_t *)calloc(mdl.n_layers, sizeof(layer_ctx_t));
    int n_gqa = 0;
    for (int l = 0; l < mdl.n_layers; l++) {
        wubu_layer_t *ly = &mdl.layers[l];
        if (ly->is_ssm) {
            lc[l].is_gqa = false;
            lc[l].w.ssm = ly->ssm;
            // Allocate SSM states (carried across steps)
            lc[l].ssm_state = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
            lc[l].conv_state = (float *)calloc((CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
            if (use_gpu) {
                gpu_load_ssm_layer(ctx, l, &lc[l].gpu_ssm_w, st);
                cudaMalloc(&lc[l].d_ssm_state, SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
                cudaMemset(lc[l].d_ssm_state, 0, SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
                cudaMalloc(&lc[l].d_conv_state, (CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
                cudaMemset(lc[l].d_conv_state, 0, (CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
                // Allocate decode scratch (B=1,T=1)
                size_t ssz = gpu_ssm_scratch_needed(1, 1);
                cudaMalloc(&lc[l].d_scratch_decode, ssz * sizeof(float));
                cudaStreamSynchronize(st);
            }
        } else {
            lc[l].is_gqa = true;
            n_gqa++;
            lc[l].w.gqa = ly->gqa;
            if (use_gpu) {
                gpu_load_gqa_layer(ctx, l, &lc[l].gpu_gqa_w, st);
                int q_dim_x2 = GQA_Q_HEADS * GQA_HEAD_DIM * 2;
                int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
                int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
                cudaMalloc(&lc[l].d_Q_full, q_dim_x2 * sizeof(float));
                cudaMalloc(&lc[l].d_K, kv_dim * sizeof(float));
                cudaMalloc(&lc[l].d_V, kv_dim * sizeof(float));
                cudaMalloc(&lc[l].d_attn_out, q_dim * sizeof(float));
                cudaStreamSynchronize(st);
            }
        }
        lc[l].moe = false;
    }
    printf("GQA: %d / %d layers | SSM: %d layers\n", n_gqa, mdl.n_layers, mdl.n_layers - n_gqa);

    // ---- Init KV caches ----
    int max_cache = 1024; // start small, grow on demand
    for (int l = 0; l < mdl.n_layers; l++) {
        if (lc[l].is_gqa) {
            if (!kv_init(&lc[l].kv, max_cache, GQA_KV_DIM)) return 1;
        }
    }

    // ---- MoE setup ----
    if (moe_on) printf("MoE setup: %d layers...\n", mdl.n_layers); fflush(stdout);
    if (moe_on) {
        for (int l = 0; l < mdl.n_layers; l++) {
            lmoe_init(&lc[l].lm);
            char nm[256];
            snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", l);
            if (!gguf_find_tensor(ctx, nm)) continue;
            lc[l].moe = true;
            snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_exps.weight", l);
            gguf_tensor_info *t = gguf_find_tensor(ctx, nm);
            if (!t) { lc[l].moe = false; continue; }
            lc[l].lm.ty_ge = t->ggml_type;
            lc[l].lm.raw_sz = gguf_raw_size(t->ggml_type, (int64_t)D_MODEL * D_FF);
            lc[l].lm.raw_sz_d = gguf_raw_size(t->ggml_type, (int64_t)D_FF * D_MODEL);
            lc[l].lm.q_gate = (const uint8_t *)ctx->data_blob + t->data_offset;

            snprintf(nm, sizeof(nm), "blk.%d.ffn_up_exps.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) lc[l].lm.q_up = (const uint8_t *)ctx->data_blob + t->data_offset;

            snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) {
                lc[l].lm.q_down = (const uint8_t *)ctx->data_blob + t->data_offset;
                lc[l].lm.ty_gd = t->ggml_type;
            }

            snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", l);
            t = gguf_find_tensor(ctx, nm);
            lc[l].lm.ty_gi = t->ggml_type;
            lc[l].lm.router = (float *)malloc(D_MODEL * N_EXPERTS * sizeof(float));
            gguf_dequantize((const uint8_t *)ctx->data_blob + t->data_offset,
                            t->ggml_type, D_MODEL * N_EXPERTS, lc[l].lm.router);

            snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_shexp.weight", l);
            t = gguf_find_tensor(ctx, nm);
            if (t) {
                lc[l].lm.ty_gs = t->ggml_type;
                int64_t sn = (int64_t)D_MODEL * SHARED_D_FF;
                int64_t sd = (int64_t)SHARED_D_FF * D_MODEL;
                lc[l].lm.sh_gate = (float *)malloc(sn * sizeof(float));
                lc[l].lm.sh_up = (float *)malloc(sn * sizeof(float));
                lc[l].lm.sh_down = (float *)malloc(sd * sizeof(float));
                gguf_dequantize((const uint8_t *)ctx->data_blob + t->data_offset,
                                t->ggml_type, sn, lc[l].lm.sh_gate);
                snprintf(nm, sizeof(nm), "blk.%d.ffn_up_shexp.weight", l);
                t = gguf_find_tensor(ctx, nm);
                if (t) gguf_dequantize((const uint8_t *)ctx->data_blob + t->data_offset,
                                       t->ggml_type, sn, lc[l].lm.sh_up);
                snprintf(nm, sizeof(nm), "blk.%d.ffn_down_shexp.weight", l);
                t = gguf_find_tensor(ctx, nm);
                if (t) gguf_dequantize((const uint8_t *)ctx->data_blob + t->data_offset,
                                       t->ggml_type, sd, lc[l].lm.sh_down);
                snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp_shexp.weight", l);
                t = gguf_find_tensor(ctx, nm);
                if (t) {
                    lc[l].lm.sh_gate_proj = (float *)malloc(D_MODEL * sizeof(float));
                    gguf_dequantize((const uint8_t *)ctx->data_blob + t->data_offset,
                                    t->ggml_type, D_MODEL, lc[l].lm.sh_gate_proj);
                }
            }
        }
    }

    // ---- Encode ----
    int *pids = (int *)malloc(65536 * sizeof(int));
    if (!pids) { fprintf(stderr, "Failed to allocate pids\\n"); return 1; }
    int np;
    
    if (chat_mode) {
        // Build chat template by manually injecting special token IDs
        // Token IDs (from vocab): <|im_start|>=248045, <|im_end|>=248046, <think>=248068
        // Newline encodes as token 198 in Qwen3.6 BPE
        const int IM_START = 248045;
        const int IM_END   = 248046;
        const int THINK    = 248068;
        const int NL_TOKEN = 198;
        
        int *pids_tmp = (int *)malloc(65536 * sizeof(int));
        if (!pids_tmp) { free(pids); fprintf(stderr, "Failed\\n"); return 1; }
        int pos = 0;
        
        // BOS
        pids_tmp[pos++] = tok.bos_id; // 248044
        
        // <|im_start|>system\nYou are a helpful assistant.
        pids_tmp[pos++] = IM_START;
        {   int n = wubu_tokenizer_encode(&tok, "system\nYou are a helpful assistant.",
                                          pids_tmp + pos, 65535 - pos);
            if (n <= 0) { fprintf(stderr, "Failed to encode system msg\n"); return 1; }
            pos += n;
        }
        
        // <|im_end|>\n<|im_start|>user\n
        pids_tmp[pos++] = IM_END;
        pids_tmp[pos++] = NL_TOKEN;
        pids_tmp[pos++] = IM_START;
        {   int n = wubu_tokenizer_encode(&tok, "user\n",
                                          pids_tmp + pos, 65535 - pos);
            if (n <= 0) { fprintf(stderr, "Failed to encode role\\n"); return 1; }
            pos += n;
        }
        
        // [USER_PROMPT]
        {   int n = wubu_tokenizer_encode(&tok, prompt,
                                          pids_tmp + pos, 65535 - pos);
            if (n <= 0) { fprintf(stderr, "Failed to encode prompt\n"); return 1; }
            pos += n;
        }
        
        // <|im_end|>\n<|im_start|>assistant\n<think>\n
        pids_tmp[pos++] = IM_END;
        pids_tmp[pos++] = NL_TOKEN;
        pids_tmp[pos++] = IM_START;
        {   int n = wubu_tokenizer_encode(&tok, "assistant\n",
                                          pids_tmp + pos, 65535 - pos);
            if (n <= 0) { fprintf(stderr, "Failed to encode assistant prefix\n"); return 1; }
            pos += n;
        }
        pids_tmp[pos++] = THINK;
        pids_tmp[pos++] = NL_TOKEN;
        
        np = pos;
        memcpy(pids, pids_tmp, np * sizeof(int));
        free(pids_tmp);
        printf("Chat prompt: %d tok\\n", np);
        if (verb) {
            printf("  Template tokens:");
            for (int i = 0; i < np && i < 20; i++) printf(" %d", pids[i]);
            printf("%s\\n", np > 20 ? " ..." : "");
        }
        // Debug: print first 10 token strings
        if (verb) {
            char dbuf[256];
            for (int i = 0; i < np && i < 10; i++) {
                wubu_tokenizer_decode(&tok, pids + i, 1, dbuf, 255);
                printf("  tok[%d] = '%s'\\n", pids[i], dbuf);
            }
        }
    } else {
        np = wubu_tokenizer_encode(&tok, prompt, pids, 65536);
        if (np <= 0) { fprintf(stderr, "Failed to encode prompt\\n"); wubu_tokenizer_free(&tok); return 1; }
        // BOS skipped — add_bos_token=false in GGUF metadata. Use env ADD_BOS=1 to force.
        if (getenv("ADD_BOS")) {
            memmove(pids + 1, pids, np * sizeof(int));
            pids[0] = tok.bos_id; np++;
        }
        printf("Prompt: %d tok (BOS: %s)\n", np, getenv("ADD_BOS")?"yes":"no");
    }

    // ---- Allocate forward buffers ----
    float *x = (float *)malloc(np * D_MODEL * sizeof(float));
    float *normed = (float *)malloc(np * D_MODEL * sizeof(float));
    float *attn = (float *)malloc(np * D_MODEL * sizeof(float));
    float *residual = (float *)malloc(np * D_MODEL * sizeof(float));
    float *ffn = (float *)malloc(np * D_MODEL * sizeof(float));

    // ---- Phase 1: Prefill ----
    printf("\n--- Phase 1: Prefill (%d tok) ---\n", np);
    fflush(stdout);
    double t_prefill = now_sec();
    printf("  Prefill start: %.3f s since prog start\n", t_prefill - T0); fflush(stdout);

    // Embed all prompt tokens
    for (int i = 0; i < np; i++) {
        int id = pids[i];
        if (id >= 0 && id < vs)
            memcpy(x + i * D_MODEL, embd + id * D_MODEL, D_MODEL * sizeof(float));
        else
            memset(x + i * D_MODEL, 0, D_MODEL * sizeof(float));
    }
    memcpy(residual, x, np * D_MODEL * sizeof(float));
    { const char *de = getenv("DUMP_EMBED"); if (de) { FILE *f = fopen(de, "wb"); if (f) { fwrite(residual, sizeof(float), np * D_MODEL, f); fclose(f); } } }
    // DEBUG: always dump embedding for comparison
    { FILE *f = fopen("/tmp/debug_embed_gpu_mode.bin", "wb"); if (f) { fwrite(residual, sizeof(float), np * D_MODEL, f); fclose(f); } }

    // Debug: embedding stats
    { float esum=0, emax=-1e30, emin=1e30;
      for (int i = 0; i < np * D_MODEL; i++) { esum += fabsf(x[i]); if(x[i]>emax)emax=x[i]; if(x[i]<emin)emin=x[i]; }
      printf("  Embd stats: mean|%.4f max|%.4f min|%.4f (norm=%.4f)\\n", esum/(np*D_MODEL), emax, emin,
             sqrtf(esum*esum/(np*D_MODEL)));
    }
    printf("  Starting layer loop...\\n"); fflush(stdout);

    const char *ml_env = getenv("MAX_LAYERS");
    int max_layers = ml_env ? atoi(ml_env) : mdl.n_layers;
    if (max_layers < 0 || max_layers > mdl.n_layers) max_layers = mdl.n_layers;
    printf("  Running %d / %d layers (MAX_LAYERS=%s)\n", max_layers, mdl.n_layers, ml_env ? ml_env : "all"); fflush(stdout);

    for (int l = 0; l < max_layers; l++) {
        printf("LL %d at %.1f s\n", l, now_sec() - t_prefill); fflush(stdout);
        double tl = now_sec();
        layer_ctx_t *ly = &lc[l];

        bool is_ssm = mdl.layers[l].is_ssm;
        wubu_rms_norm(1, np, D_MODEL, residual, mdl.layers[l].attn_norm_weight, 1e-6f, normed);
        // DEBUG: dump normed + weight for layer 0
        if (l == 0) {
            FILE *f = fopen("/tmp/debug_layer0_normed.bin", "wb");
            if (f) { fwrite(normed, sizeof(float), np * D_MODEL, f); fclose(f); }
            f = fopen("/tmp/debug_layer0_attn_norm_weight.bin", "wb");
            if (f) { fwrite(mdl.layers[l].attn_norm_weight, sizeof(float), D_MODEL, f); fclose(f); }
        }

        if (ly->is_gqa) {
            printf("L%d GQA start\n", l); fflush(stdout);

            if (use_gpu) {
                // GPU GQA prefill: upload, run forward + KV extract
                int kv_dim = GQA_KV_DIM;
                float *K_cache = (float *)malloc(np * kv_dim * sizeof(float));
                float *V_cache = (float *)malloc(np * kv_dim * sizeof(float));
                float *gpu_out = (float *)malloc(np * D_MODEL * sizeof(float));

                // Allocate scratch buffer for input (reuse d_scratch)
                float *d_scratch_prefill;
                size_t scratch_nf = (size_t)np * D_MODEL;
                cudaMalloc(&d_scratch_prefill, scratch_nf * sizeof(float));

                gpu_gqa_prefill(ch, st, normed, 1, np,
                    &ly->gpu_gqa_w, d_sincos,
                    ly->d_Q_full, ly->d_K, ly->d_V,
                    d_scratch_prefill,
                    gpu_out,
                    K_cache, V_cache);

                memcpy(attn, gpu_out, np * D_MODEL * sizeof(float));

                // Populate KV cache: RMSNorm K + RoPE on CPU
                float *K_norm_cache = (float *)malloc(np * kv_dim * sizeof(float));
                memcpy(K_norm_cache, K_cache, np * kv_dim * sizeof(float));
                wubu_rms_norm(1, np * GQA_KV_HEADS, GQA_HEAD_DIM,
                             K_norm_cache, ly->w.gqa.attn_k_norm_weight, 1e-6f, K_norm_cache);
                for (int s = 0; s < np; s++)
                    apply_rotary_to_buf(K_norm_cache + s * kv_dim, GQA_KV_HEADS, s, rope_sc);
                kv_append(&ly->kv, K_norm_cache, V_cache, np);

                cudaFree(d_scratch_prefill);
                free(K_cache); free(V_cache); free(K_norm_cache); free(gpu_out);
                printf("L%d GQA_GPU_DONE\n", l); fflush(stdout);
            } else {
                // CPU GQA prefill (original)
                int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
                int kv_dim = GQA_KV_DIM;

                printf("L%d alloc\n", l); fflush(stdout);
                float *Q = (float *)malloc(np * q_dim * sizeof(float));
                float *K = (float *)malloc(np * kv_dim * sizeof(float));
                float *V = (float *)malloc(np * kv_dim * sizeof(float));
                float *K_norm = (float *)malloc(np * kv_dim * sizeof(float));
                float *gate_buf = (float *)malloc(np * q_dim * sizeof(float));
                float *attn_out = (float *)calloc(np * q_dim, sizeof(float));
                printf("L%d alloc done: %p %p %p\n", l, (void*)Q, (void*)K, (void*)V); fflush(stdout);

                    // DEBUG: dump normed input for L3
                if (l == 3 && np == 1) {
                    FILE *f = fopen("/tmp/debug_gqa_normed_input.bin", "wb");
                    if (f) { fwrite(normed, sizeof(float), np * D_MODEL, f); fclose(f); }
                }
                #pragma omp parallel for if(np > 1)
                for (int s = 0; s < np; s++) {
                    const float *xs = normed + s * D_MODEL;
                    // Q
                    for (int j = 0; j < q_dim; j++) {
                        double sum = 0.0;
                        for (int i = 0; i < D_MODEL; i++)
                            sum += (double)xs[i] * (double)ly->w.gqa.attn_q_weight[i + j * D_MODEL];
                        Q[s * q_dim + j] = (float)sum;
                    }
                    // gate
                    for (int j = 0; j < q_dim; j++) {
                        double sum = 0.0;
                        for (int i = 0; i < D_MODEL; i++)
                            sum += (double)xs[i] * (double)ly->w.gqa.attn_q_weight[i + (j + q_dim) * D_MODEL];
                        gate_buf[s * q_dim + j] = (float)sum;
                    }
                    // K
                    for (int j = 0; j < kv_dim; j++) {
                        double sum = 0.0;
                        for (int i = 0; i < D_MODEL; i++)
                            sum += (double)xs[i] * (double)ly->w.gqa.attn_k_weight[i + j * D_MODEL];
                        K[s * kv_dim + j] = (float)sum;
                    }
                    // V
                    for (int j = 0; j < kv_dim; j++) {
                        double sum = 0.0;
                        for (int i = 0; i < D_MODEL; i++)
                            sum += (double)xs[i] * (double)ly->w.gqa.attn_v_weight[i + j * D_MODEL];
                        V[s * kv_dim + j] = (float)sum;
                    }
                }
                // DEBUG: dump L3 Q, K, V, gate raw
                if (l == 3 && np == 1) {
                    FILE *f;
                    f = fopen("/tmp/debug_gqa_q_raw.bin", "wb"); if(f){fwrite(Q, sizeof(float), np*q_dim, f); fclose(f);}
                    f = fopen("/tmp/debug_gqa_k_raw.bin", "wb"); if(f){fwrite(K, sizeof(float), np*kv_dim, f); fclose(f);}
                    f = fopen("/tmp/debug_gqa_v_raw.bin", "wb"); if(f){fwrite(V, sizeof(float), np*kv_dim, f); fclose(f);}
                    f = fopen("/tmp/debug_gqa_gate_raw.bin", "wb"); if(f){fwrite(gate_buf, sizeof(float), np*q_dim, f); fclose(f);}
                }

                // RMSNorm Q and K
                { float q_raw_sum=0,q_raw_mx=-1e30,q_raw_mn=1e30;
                  for(int i=0;i<np*q_dim;i++){q_raw_sum+=Q[i];if(Q[i]>q_raw_mx)q_raw_mx=Q[i];if(Q[i]<q_raw_mn)q_raw_mn=Q[i];}
                  printf("  Q_raw: mean=%.4f range=[%.4f,%.4f]\n",q_raw_sum/(np*q_dim),q_raw_mn,q_raw_mx); }
                memcpy(K_norm, K, np * kv_dim * sizeof(float));
                wubu_rms_norm(1, np * GQA_Q_HEADS, GQA_HEAD_DIM, Q, ly->w.gqa.attn_q_norm_weight, 1e-6f, Q);
                wubu_rms_norm(1, np * GQA_KV_HEADS, GQA_HEAD_DIM, K_norm, ly->w.gqa.attn_k_norm_weight, 1e-6f, K_norm);

                // DEBUG: dump Q/K stats
                { float q_sum=0,q_mx=-1e30,q_mn=1e30,q_rms_sum=0,k_sum=0,k_mx=-1e30,k_mn=1e30,k_rms_sum=0;
                  for(int i=0;i<np*q_dim;i++){q_sum+=Q[i];q_rms_sum+=Q[i]*Q[i];if(Q[i]>q_mx)q_mx=Q[i];if(Q[i]<q_mn)q_mn=Q[i];}
                  for(int i=0;i<np*kv_dim;i++){k_sum+=K_norm[i];k_rms_sum+=K_norm[i]*K_norm[i];if(K_norm[i]>k_mx)k_mx=K_norm[i];if(K_norm[i]<k_mn)k_mn=K_norm[i];}
                  printf("  Q_norm: mean=%.4f rms=%.4f range=[%.4f,%.4f] | K_norm: mean=%.4f rms=%.4f range=[%.4f,%.4f]\n",
                         q_sum/(np*q_dim),sqrtf(q_rms_sum/(np*q_dim)),q_mn,q_mx,
                         k_sum/(np*kv_dim),sqrtf(k_rms_sum/(np*kv_dim)),k_mn,k_mx); }

                // Apply RoPE to Q and K (in-place) for each position
                for (int s = 0; s < np; s++) {
                    apply_rotary_to_buf(Q + s * q_dim, GQA_Q_HEADS, s, rope_sc);
                    apply_rotary_to_buf(K_norm + s * kv_dim, GQA_KV_HEADS, s, rope_sc);
                }

                // Attention for each position (causal mask)
                float scale = 1.0f / sqrtf((float)GQA_HEAD_DIM);
                #pragma omp parallel for if(np > 1)
                for (int s = 0; s < np; s++) {
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

                // Gate
                for (int i = 0; i < np * q_dim; i++)
                    attn_out[i] *= 1.0f / (1.0f + expf(-gate_buf[i]));
                printf("  GQA_GATE_DONE at %.1f s\n", now_sec() - t_prefill); fflush(stdout);

                // Output projection
                memset(attn, 0, np * D_MODEL * sizeof(float));
                printf("  OUTPROJ START at %.1f s\n", now_sec() - t_prefill); fflush(stdout);
                #pragma omp parallel for collapse(2)
                for (int s = 0; s < np; s++) {
                    for (int j = 0; j < D_MODEL; j++) {
                        double sum = 0.0;
                        const float *in = attn_out + s * q_dim;
                        for (int i = 0; i < q_dim; i++)
                            sum += (double)in[i] * (double)ly->w.gqa.attn_output_weight[i + j * q_dim];
                        attn[s * D_MODEL + j] = (float)sum;
                    }
                }

                // DEBUG: dump GQA output before residual add (only L3)
                { char fn[256]; snprintf(fn,256,"/tmp/debug_gqa_l%d.bin",l);
                  FILE *f_debug = fopen(fn, "wb"); if(f_debug){fwrite(attn, sizeof(float), np*D_MODEL, f_debug);fclose(f_debug);}}
                // Populate KV cache with K_norm and V
                kv_append(&ly->kv, K_norm, V, np);
                printf("  GQA_DONE at %.1f s\n", now_sec() - t_prefill); fflush(stdout);

                free(Q); free(K); free(V); free(K_norm); free(gate_buf); free(attn_out);
            }
        } else {
            // SSM forward: full prefill, save final state
            printf("L%d SSM: B=1 T=%d\n", l, np); fflush(stdout);
            if (use_gpu) {
                float *d_scratch_prefill;
                size_t snf = gpu_ssm_scratch_needed(1, np);
                cudaMalloc(&d_scratch_prefill, snf * sizeof(float));

                gpu_ssm_prefill(ch, st, normed, 1, np,
                    &ly->gpu_ssm_w,
                    ly->d_ssm_state, ly->d_conv_state,
                    d_scratch_prefill,
                    attn);

                cudaFree(d_scratch_prefill);
                // DEBUG: dump GPU SSM output for layer 0
                if (l == 0) {
                    FILE *f = fopen("/tmp/debug_gpu_ssm_out.bin", "wb");
                    if (f) { fwrite(attn, sizeof(float), np * D_MODEL, f); fclose(f); }
                }
                printf("L%d SSM_GPU_DONE\\n", l); fflush(stdout);
            } else {
                wubu_ssm_forward(normed, 1, np, &ly->w.ssm,
                                 ly->ssm_state, ly->conv_state, attn, NULL, NULL);
                // DEBUG: dump SSM output for layer 0
                if (l == 0) {
                    FILE *f = fopen("/tmp/debug_layer0_ssm_out.bin", "wb");
                    if (f) { fwrite(attn, sizeof(float), np * D_MODEL, f); fclose(f); }
                }
                printf("L%d SSM done\\n", l); fflush(stdout);
            }
        }

        // NaN check
        if (verb) {
            int has_nan = 0;
            for (int i = 0; i < np * D_MODEL; i++)
                if (isnan(attn[i]) || isinf(attn[i])) { has_nan = 1; break; }
            if (has_nan) printf("  L%d NaN\n", l);
        }

        // Debug: dump layer 0 hidden state stats
        if (l == 0) {
            float sum=0, maxv=-1e30, minv=1e30;
            for (int i = 0; i < np * D_MODEL; i++) {
                sum += fabsf(attn[i]);
                if (attn[i] > maxv) maxv = attn[i];
                if (attn[i] < minv) minv = attn[i];
            }
            printf("  L0 attn_out: mean|%.4f max|%.4f min|%.4f\n", sum/(np*D_MODEL), maxv, minv);
        }

        // Residual
        for (int i = 0; i < np * D_MODEL; i++) residual[i] += attn[i];

        // Per-layer dump
        maybe_dump_layer(l, residual, np);

        // Dump layer 0 residual stats
        if (l == 0) {
            float minv = 1e30, maxv = -1e30, sum = 0, sumsq = 0;
            for (int i = 0; i < np * D_MODEL; i++) {
                if (residual[i] < minv) minv = residual[i];
                if (residual[i] > maxv) maxv = residual[i];
                sum += residual[i];
                sumsq += residual[i] * residual[i];
            }
            printf("  L%d residual: min=%.4f max=%.4f mean=%.4f rms=%.4f\n",
                   l, minv, maxv, sum/(np*D_MODEL), sqrtf(sumsq/(np*D_MODEL)));
        }

        // Post-attention RMSNorm
        printf("  NORMB at %.1f s\n", now_sec() - t_prefill); fflush(stdout);
        wubu_rms_norm(1, np, D_MODEL, residual, mdl.layers[l].post_attn_norm_weight, 1e-6f, normed);
        printf("  NORM AFTER at %.1f s\n", now_sec() - t_prefill); fflush(stdout);

        // MoE (lazy prefill)
        if (moe_on && lc[l].moe && (moe_max_l == 0 || l < moe_max_l)) {
            lazy_moe_decode(normed, 1, np,
                            lc[l].lm.q_gate, lc[l].lm.q_up, lc[l].lm.q_down,
                            &lc[l].lm, ffn);
            // DEBUG: dump MoE output for layer 0
            if (l == 0) {
                FILE *f = fopen("/tmp/debug_layer0_moe_out.bin", "wb");
                if (f) { fwrite(ffn, sizeof(float), np * D_MODEL, f); fclose(f); }
            }
         } else {
            printf("  MOE_MEMCPY at %.1f s\n", now_sec() - t_prefill); fflush(stdout);
            memcpy(ffn, normed, np * D_MODEL * sizeof(float));
            printf("  MOE_MEMCPY_DONE at %.1f s\n", now_sec() - t_prefill); fflush(stdout);
        }

        for (int i = 0; i < np * D_MODEL; i++) residual[i] += ffn[i];

        // Post-MoE dump (same naming as reference's post_moe)
        if (getenv("DUMP_LAYER_DIR")) maybe_dump_layer(100 + l, residual, np);

        if (verb) printf("  L%d: %.3f ms\n", l, (now_sec() - tl) * 1000);
        if (l < 5 || l == mdl.n_layers-1) printf("  L%d: %.3f s\n", l, now_sec() - t_prefill);
    }

    // Final RMSNorm
    printf("  FINAL_NORM at %.1f s\n", now_sec() - t_prefill); fflush(stdout);
    if (mdl.norm_weight) {
        // DEBUG: dump residual before norm
        { FILE *f_debug = fopen("/tmp/debug_res_before_norm.bin", "wb"); if(f_debug){fwrite(residual, sizeof(float), np*D_MODEL, f_debug);fclose(f_debug);}}
        // DEBUG: dump norm_weight
        { FILE *f_debug = fopen("/tmp/debug_norm_weight.bin", "wb"); if(f_debug){fwrite(mdl.norm_weight, sizeof(float), D_MODEL, f_debug);fclose(f_debug);}}
        wubu_rms_norm(1, np, D_MODEL, residual, mdl.norm_weight, 1e-6f, normed);
        // DEBUG: dump normed output
        { FILE *f_debug = fopen("/tmp/debug_normed.bin", "wb"); if(f_debug){fwrite(normed, sizeof(float), np*D_MODEL, f_debug);fclose(f_debug);}}
        memcpy(residual, normed, np * D_MODEL * sizeof(float));
    }
    printf("  FINAL_NORM_DONE at %.1f s\n", now_sec() - t_prefill); fflush(stdout);

    // Dump final hidden state
    maybe_dump_hidden(residual, np);

    // Output projection for last token
    const float *h_last = residual + (np - 1) * D_MODEL;
    { float mn=1e30,mx=-1e30,sum=0; for(int k=0;k<D_MODEL;k++){mn=fminf(mn,h_last[k]);mx=fmaxf(mx,h_last[k]);sum+=h_last[k];}
      printf("  h_last: mean=%.4f max=%.4f min=%.4f rms=%.4f nonzero=%d\n", sum/D_MODEL,mx,mn,sqrtf(sum/D_MODEL),mx>1e-10?1:0);
      // Also dump first 5 output weights at k=0
      printf("  out_w[0..4]: %.6e %.6e %.6e %.6e %.6e\n",
             mdl.output_weight[0], mdl.output_weight[1], mdl.output_weight[2], mdl.output_weight[3], mdl.output_weight[4]);
      printf("  out_w[0+1*vs..0+5*vs]: %.6e %.6e %.6e %.6e %.6e\n",
             mdl.output_weight[(int64_t)0+(int64_t)0*vs], mdl.output_weight[(int64_t)0+(int64_t)1*vs],
             mdl.output_weight[(int64_t)0+(int64_t)2*vs], mdl.output_weight[(int64_t)0+(int64_t)3*vs],
             mdl.output_weight[(int64_t)0+(int64_t)4*vs]); }
    float *logits = (float *)malloc(vs * sizeof(float));
    if (use_gpu) {
        printf("  GPU proj start\n"); fflush(stdout);
        cudaMemcpyAsync(d_hid, h_last, D_MODEL * sizeof(float), cudaMemcpyHostToDevice, st);
        gpu_output_projection(ch, st, d_hid, 1, 1, d_out_w, vs, d_log);
        cudaMemcpyAsync(logits, d_log, vs * sizeof(float), cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        printf("  GPU proj done\\n"); fflush(stdout);
    } else if (mdl.output_weight) {
        #pragma omp parallel for
        for (int j = 0; j < vs; j++) {
            double sum = 0.0;
            for (int k = 0; k < D_MODEL; k++)
                sum += (double)h_last[k] * (double)mdl.output_weight[(int64_t)j * D_MODEL + k];
            logits[j] = (float)sum;
        }
    } else {
        memcpy(logits, h_last, D_MODEL * sizeof(float));
    }
    
    // If DUMP_LOGITS env var set, write raw logits to file
    const char *dump_path = getenv("DUMP_LOGITS");
    if (dump_path) {
        FILE *f = fopen(dump_path, "wb");
        if (f) {
            fwrite(logits, sizeof(float), vs, f);
            fclose(f);
            printf("  Logits dumped to %s\\n", dump_path);
        }
    }
    
    // Sample first token
    int tid = sample(logits, vs, temperature, samp_top_k, top_p, pids, np, rep_penalty);
    pids[np] = tid;
    int n_total = np + 1;

    // Debug: print top-5 logits
    { int n5 = vs < 5 ? vs : 5; int top[5] = {0}; float topv[5] = {-1e30,-1e30,-1e30,-1e30,-1e30};
      for (int j = 0; j < vs; j++) {
        if (logits[j] > topv[n5-1]) {
          topv[n5-1] = logits[j]; top[n5-1] = j;
          for (int k = n5-2; k >= 0; k--) { if (topv[k] < topv[k+1]) {
            float tv = topv[k]; int ti = top[k]; topv[k] = topv[k+1]; top[k] = top[k+1]; topv[k+1] = tv; top[k+1] = ti;
          } }
        } }
      printf("\nTop-5 tokens: ");
      for (int k = 0; k < n5; k++) {
        char buf[256] = {0}; wubu_tokenizer_decode(&tok, top+k, 1, buf, 255);
        printf(" [%d]='%s'(%.2f)", top[k], buf, topv[k]);
      }
      printf("\n");
    }

    // Print prompt + first generated token
    char out_buf[1048576];
    int prev_out_len = 0;
    int cur_out_len = wubu_tokenizer_decode(&tok, pids, n_total, out_buf, 1048576);
    if (cur_out_len > 0) { out_buf[cur_out_len] = '\0'; printf("%s", out_buf); fflush(stdout); }
    prev_out_len = cur_out_len;

    double prefill_time = now_sec() - t_prefill;
    printf("\nPrefill: %.2f s (%.0f tok/s)\n", prefill_time, np / prefill_time);
    printf("--- Phase 2: Decode ---\n");

    // ---- Phase 2: Decode (token-by-token with KV cache) ----
    int gen = 1; // already generated 1 token
    double t_gen = now_sec();
    float *x_step = (float *)malloc(D_MODEL * sizeof(float));
    float *out_step = (float *)malloc(D_MODEL * sizeof(float));

    // Grow KV caches if needed for full generation
    if (max_cache < n_total + max_tok) {
        // Can't grow easily — for now just warn
        printf("WARN: KV cache may overflow (max=%d, need=%d)\n", max_cache, n_total + max_tok);
    }

    // Decode loop
    int last_token = pids[np]; // the first generated token
    while (gen < max_tok && !stop) {
        // Embed last generated token
        int id = last_token;
        if (id >= 0 && id < vs)
            memcpy(x_step, embd + id * D_MODEL, D_MODEL * sizeof(float));
        else
            memset(x_step, 0, D_MODEL * sizeof(float));

        // We need to track the residual from the last step's output
        // For the first decode step, x_step IS the residual
        // For subsequent steps, we carry the residual forward
        // Actually, we need the full residual chain, not just the new embedding.
        // This is the tricky part — we need to run the full layer stack for 1 token.

        // For now, use the simple approach: run full forward with all tokens in context
        // (same as v1, but using KV cache for GQA layers in forward)
        // Actually, this is getting complex. Let me simplify:

        // Build full input
        // ... but that's O(T) per step again for embed and norm.

        // The correct approach: run 1 token through all layers, with:
        // - GQA: uses KV cache (already populated)
        // - SSM: carries state
        // - Residual: starts at the new token's embedding

        // But for each layer, the residual input is the output of the previous layer.
        // For the first layer, residual = new_token_embedding.
        // For subsequent layers, residual = prev_layer_output.
        // So we just chain 1 token through all layers.

        memset(residual, 0, D_MODEL * sizeof(float));
        memcpy(residual, x_step, D_MODEL * sizeof(float));

        for (int l = 0; l < mdl.n_layers; l++) {
            layer_ctx_t *ly = &lc[l];

            // Pre-attention RMSNorm (1 token)
            wubu_rms_norm(1, 1, D_MODEL, residual, mdl.layers[l].attn_norm_weight, 1e-6f, normed);

            if (ly->is_gqa) {
                if (use_gpu) {
                    int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
                    int kv_dim = GQA_KV_DIM;
                    float h_Q_full[8192], h_K[512], h_V[512];

                    // Upload x_step to GPU
                    cudaMemcpyAsync(ly->d_Q_full, normed, D_MODEL * sizeof(float),
                                    cudaMemcpyHostToDevice, st);
                    cudaStreamSynchronize(st);

                    // Projections on GPU
                    gpu_gqa_decode_step(ch, st, ly->d_Q_full,
                        &ly->gpu_gqa_w,
                        ly->d_Q_full, ly->d_K, ly->d_V,
                        ly->d_attn_out,
                        h_Q_full, h_K, h_V);

                    // CPU: RMSNorm Q + RoPE
                    float *q_norm = h_Q_full; // reuse first q_dim
                    memcpy(q_norm, h_Q_full, q_dim * sizeof(float));
                    wubu_rms_norm(1, GQA_Q_HEADS, GQA_HEAD_DIM, q_norm,
                                  ly->w.gqa.attn_q_norm_weight, 1e-6f, q_norm);
                    apply_rotary_to_buf(q_norm, GQA_Q_HEADS, ly->kv.current_T, rope_sc);

                    // CPU: RMSNorm K + RoPE
                    float k_norm[512];
                    memcpy(k_norm, h_K, kv_dim * sizeof(float));
                    wubu_rms_norm(1, GQA_KV_HEADS, GQA_HEAD_DIM, k_norm,
                                  ly->w.gqa.attn_k_norm_weight, 1e-6f, k_norm);
                    apply_rotary_to_buf(k_norm, GQA_KV_HEADS, ly->kv.current_T, rope_sc);

                    // CPU: KV cache append
                    kv_append(&ly->kv, k_norm, h_V, 1);
                    int new_T = ly->kv.current_T;

                    // CPU: Attention
                    float *gate = h_Q_full + q_dim; // second half of Q_full
                    float attn_out[4096];
                    memset(attn_out, 0, q_dim * sizeof(float));
                    float scale = 1.0f / sqrtf((float)GQA_HEAD_DIM);
                    for (int h_q = 0; h_q < GQA_Q_HEADS; h_q++) {
                        int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS);
                        const float *qv = q_norm + h_q * GQA_HEAD_DIM;
                        float *out = attn_out + h_q * GQA_HEAD_DIM;

                        float mx = -1e30f, sum_exp = 0.0f;
                        for (int t = 0; t < new_T; t++) {
                            const float *kv = ly->kv.h_k + t * kv_dim + h_kv * GQA_HEAD_DIM;
                            float s = 0.0f;
                            for (int i = 0; i < GQA_HEAD_DIM; i++) s += qv[i] * kv[i];
                            s *= scale;
                            if (t == 0 || s > mx) mx = s;
                        }
                        for (int t = 0; t < new_T; t++) {
                            const float *kv = ly->kv.h_k + t * kv_dim + h_kv * GQA_HEAD_DIM;
                            float s = 0.0f;
                            for (int i = 0; i < GQA_HEAD_DIM; i++) s += qv[i] * kv[i];
                            s = expf(s * scale - mx);
                            sum_exp += s;
                        }
                        float inv = 1.0f / (sum_exp + 1e-30f);
                        for (int t = 0; t < new_T; t++) {
                            const float *vv = ly->kv.h_v + t * kv_dim + h_kv * GQA_HEAD_DIM;
                            const float *kv = ly->kv.h_k + t * kv_dim + h_kv * GQA_HEAD_DIM;
                            float s = 0.0f;
                            for (int i = 0; i < GQA_HEAD_DIM; i++) s += qv[i] * kv[i];
                            float a = expf(s * scale - mx) * inv;
                            for (int i = 0; i < GQA_HEAD_DIM; i++) out[i] += a * vv[i];
                        }
                    }

                    // Gate
                    for (int i = 0; i < q_dim; i++)
                        attn_out[i] *= 1.0f / (1.0f + expf(-gate[i]));

                    // Upload attn_out to GPU for output projection
                    cudaMemcpyAsync(ly->d_attn_out, attn_out, q_dim * sizeof(float),
                                    cudaMemcpyHostToDevice, st);
                    cudaStreamSynchronize(st);

                    // Output projection on GPU
                    gpu_gqa_output_proj(ch, st, ly->d_attn_out, &ly->gpu_gqa_w, ly->d_K);
                    // ly->d_K reused as d_output buffer

                    // Download output
                    cudaMemcpyAsync(out_step, ly->d_K, D_MODEL * sizeof(float),
                                    cudaMemcpyDeviceToHost, st);
                    cudaStreamSynchronize(st);
                } else {
                    gqa_kv_decode(normed, &ly->w.gqa, &ly->kv, out_step);
                }
            } else {
                if (use_gpu) {
                    // Upload input
                    cudaMemcpyAsync(ly->d_Q_full, normed, D_MODEL * sizeof(float),
                                    cudaMemcpyHostToDevice, st);
                    cudaStreamSynchronize(st);

                    // SSM decode: everything on GPU, state carries over
                    gpu_ssm_decode_step(ch, st, ly->d_Q_full,
                        &ly->gpu_ssm_w,
                        ly->d_ssm_state, ly->d_conv_state,
                        ly->d_attn_out, // reuse as output buffer
                        ly->d_scratch_decode);

                    // Download output
                    cudaMemcpyAsync(out_step, ly->d_attn_out, D_MODEL * sizeof(float),
                                    cudaMemcpyDeviceToHost, st);
                    cudaStreamSynchronize(st);
                } else {
                    ssm_kv_decode(normed, &ly->w.ssm, ly->ssm_state, ly->conv_state, out_step);
                }
            }

            for (int i = 0; i < D_MODEL; i++) residual[i] += out_step[i];

            // Post-attention RMSNorm
            wubu_rms_norm(1, 1, D_MODEL, residual, mdl.layers[l].post_attn_norm_weight, 1e-6f, normed);

            // MoE (lazy cache)
            if (moe_on && lc[l].moe && (moe_max_l == 0 || l < moe_max_l)) {
                lazy_moe_decode(normed, 1, 1,
                                lc[l].lm.q_gate, lc[l].lm.q_up, lc[l].lm.q_down,
                                &lc[l].lm, out_step);
            } else {
                memcpy(out_step, normed, D_MODEL * sizeof(float));
            }

            for (int i = 0; i < D_MODEL; i++) residual[i] += out_step[i];
        }

        // Final RMSNorm
        if (mdl.norm_weight) {
            wubu_rms_norm(1, 1, D_MODEL, residual, mdl.norm_weight, 1e-6f, normed);
            memcpy(residual, normed, D_MODEL * sizeof(float));
        }

        // Output projection
        if (use_gpu) {
            cudaMemcpyAsync(d_hid, residual, D_MODEL * sizeof(float), cudaMemcpyHostToDevice, st);
            gpu_output_projection(ch, st, d_hid, 1, 1, d_out_w, vs, d_log);
            cudaMemcpyAsync(logits, d_log, vs * sizeof(float), cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
        } else if (mdl.output_weight) {
            #pragma omp parallel for
            for (int j = 0; j < vs; j++) {
                double sum = 0.0;
                for (int k = 0; k < D_MODEL; k++)
                    sum += (double)residual[k] * (double)mdl.output_weight[(int64_t)j * D_MODEL + k];
                logits[j] = (float)sum;
            }
        } else {
            memcpy(logits, residual, D_MODEL * sizeof(float));
        }

        // Sample
        // Build recent tokens for repetition penalty (last 256)
        int penalty_window = n_total < 256 ? n_total : 256;
        last_token = sample(logits, vs, temperature, samp_top_k, top_p,
                            pids + n_total - penalty_window, penalty_window, rep_penalty);

        // Decode & print
        pids[n_total] = last_token;
        n_total++;
        int new_out = wubu_tokenizer_decode(&tok, pids, n_total, out_buf, 1048576);
        if (new_out > prev_out_len) {
            out_buf[new_out] = '\0';
            printf("%s", out_buf + prev_out_len);
            fflush(stdout);
            prev_out_len = new_out;
        }

        gen++;

        // EOS
        if (last_token == tok.eos_id && gen > 1) break;

        // Debug: dump top-3 logits at each step
        { int dn = 3; int dtop[3] = {0}; float dtv[3] = {-1e30,-1e30,-1e30};
          for (int j = 0; j < vs; j++) {
            if (logits[j] > dtv[dn-1]) {
              dtv[dn-1] = logits[j]; dtop[dn-1] = j;
              for (int k = dn-2; k >= 0; k--) { if (dtv[k] < dtv[k+1]) {
                float tv = dtv[k]; int ti = dtop[k]; dtv[k] = dtv[k+1]; dtop[k] = dtop[k+1]; dtv[k+1] = tv; dtop[k+1] = ti;
              } }
            } }
          char buf[256] = {0}; wubu_tokenizer_decode(&tok, dtop, 1, buf, 255);
          printf(" [step%d] top=%d '%s'(%.2f)", gen, dtop[0], buf, dtv[0]);
          fflush(stdout);
        }
    }

    double decode_time = now_sec() - t_gen;
    printf("\n\n=== Generation ===\n");
    printf("Prefill: %d tok in %.2f s (%.0f tok/s) | Out proj OMP threads=%d\n", np, prefill_time, np / prefill_time, omp_get_max_threads());
    printf("Decode:  %d tok in %.2f s (%.1f tok/s)\n", gen, decode_time, gen / decode_time);
    printf("Total:   %.2f s\n", now_sec() - T0);

    // ---- Cleanup ----
    free(pids);
    free(x); free(normed); free(attn); free(residual); free(ffn);
    free(x_step); free(out_step);
    free(logits); free(embd);
    for (int l = 0; l < mdl.n_layers; l++) {
        kv_free(&lc[l].kv);
        free(lc[l].ssm_state);
        free(lc[l].conv_state);
        lmoe_free(&lc[l].lm);
        if (use_gpu) {
            if (lc[l].is_gqa) {
                gpu_free_gqa_weights(&lc[l].gpu_gqa_w);
                cudaFree(lc[l].d_Q_full);
                cudaFree(lc[l].d_K);
                cudaFree(lc[l].d_V);
                cudaFree(lc[l].d_attn_out);
            } else {
                gpu_free_ssm_weights(&lc[l].gpu_ssm_w);
                cudaFree(lc[l].d_ssm_state);
                cudaFree(lc[l].d_conv_state);
                cudaFree(lc[l].d_scratch_decode);
            }
        }
    }
    free(lc);
    if (use_gpu) {
        gpu_free_output_weight(d_out_w);
        cudaFree(d_hid); cudaFree(d_log);
        if (d_sincos) cudaFree(d_sincos);
        cudaStreamDestroy(st);
        cublasDestroy(ch);
        cleanup_gpu();  // force device reset to release all VRAM
    }
    wubu_model_free(&mdl);
    wubu_tokenizer_free(&tok);
    free(rope_sc); rope_sc = NULL;

    printf("=== PASS ===\n");
    return 0;
}
