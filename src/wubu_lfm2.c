/* wubu_lfm2.c -- LFM2.5 hybrid loader + forward (C11, self-contained).
 * See wubu_lfm2.h for the architecture. Reuses gguf_reader's safetensors
 * parser + dequant, and wubuwizard's quantized_matmul / rmsnorm. */
#include "wubu_lfm2.h"
#include "wubu_win.h"
#include "gguf_reader.h"        /* quantized_matmul() for F32 weights */
#include "safetensors_reader.h"  /* st tensor types */
#include "wubu_safetensors_shard.h" /* multi-shard LFM2.5 loader */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <stdio.h>

/* ---- small CPU helpers ---- */
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "lfm2 oom\n"); exit(1);} return p; }

static st_ctx *g_shards[8];
static int g_nsh = 0; /* LFM2.5 multi-shard safetensors ctx (self-scanned) */

static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    float f; memcpy(&f, &u, 4); return f;
}

/* Load a BF16/BF16 safetensors tensor by exact name across opened shards
 * and dequantize to F32. Returns malloc'd F32 buffer (caller frees) or NULL. */
static float *load_bf16_f32(const char *name, int64_t *out_ne) {
    for (int s = 0; s < g_nsh; s++) {
        const st_tensor_info *t = st_find_tensor(g_shards[s], name);
        if (!t) continue;
        int64_t ne = t->n_elems;
        float *f = (float *)xmalloc((size_t)ne * sizeof(float));
        if (st_read_tensor_f32(g_shards[s], t, f, ne) != ne) { free(f); continue; }
        if (out_ne) *out_ne = ne;
        return f;
    }
    return NULL;
}

/* Plain F32 matmul: y[M,N] = x[M,K] @ W.T where W is stored [N,K] row-major
 * (PyTorch nn.Linear convention: y = x @ W^T). Self-contained correct GEMM;
 * NOT quantized_matmul's F32 path (which silently INT4-quantizes weights). */
static void matmul_f32(const float *x, const float *W, int M, int K, int N, float *y) {
    for (int i = 0; i < M; i++) {
        const float *xr = x + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            const float *wr = W + (size_t)j * K;   /* row j of W (out dim j, in dim k) */
            for (int k = 0; k < K; k++) s += xr[k] * wr[k];
            y[(size_t)i * N + j] = s;
        }
    }
}

/* RMSNorm: y = x / rms(x) * gamma ; eps baked into caller or default. */
static void rmsnorm(float *x, const float *gamma, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / n + eps);
    float inv = 1.0f / rms;
    for (int i = 0; i < n; i++) x[i] = x[i] * inv * gamma[i];
}

/* Depthwise causal conv (kernel k) over sequence T, channels C.
 * PyTorch nn.Conv1d(hidden, hidden, kernel_size=k, padding=k-1, groups=hidden)
 * computes out[t] = sum_{j} w[j] * in[t - (k-1) + j]. So weight index j
 * multiplies the sample (k-1-j) positions back; we reverse the kernel index.
 * w[C, k]; y[t,c] = sum_{j=0..k-1} w[c,(k-1)-j] * x[t-j,c] (x padded 0 for t<0). */
static void depthwise_conv(const float *x, const float *w, int T, int C, int k, float *y) {
    int last = k - 1;
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            float s = 0.0f;
            for (int j = 0; j < k; j++) {
                int tt = t - j;
                if (tt >= 0) s += w[(size_t)c * k + (last - j)] * x[(size_t)tt * C + c];
            }
            y[(size_t)t * C + c] = s;
        }
    }
}

/* GQA with RoPE + q/k layernorm. x: [T, d_model]; writes attn_out [T, d_model].
 * Uses an internal KV cache (per layer) appended across calls. */
static void lfm2_gqa(const lfm2_layer_t *L, const lfm2_model_t *m,
                     const float *x, int T, float *attn_out, float *kv_cache_layer) {
    int d = m->d_model, hd = m->head_dim;
    int nq = m->n_q_heads, nkv = m->n_kv_heads;
    int kv_dim = nkv * hd;
    int Tprev = m->kv_max_t; /* cache slots already filled (prefill offset) */
    int Ttot = Tprev + T;
    float *q = (float *)xmalloc((size_t)T * d * sizeof(float));
    float *k = (float *)xmalloc((size_t)T * kv_dim * sizeof(float));
    float *v = (float *)xmalloc((size_t)T * kv_dim * sizeof(float));
    matmul_f32(x, L->q_proj, T, d, d, q);
    matmul_f32(x, L->k_proj, T, d, kv_dim, k);
    matmul_f32(x, L->v_proj, T, d, kv_dim, v);

    /* copy new K/V into cache */
    if (kv_cache_layer) {
        memcpy(kv_cache_layer + (size_t)Tprev * kv_dim, k, (size_t)T * kv_dim * sizeof(float));
        memcpy(kv_cache_layer + m->kv_max_t * kv_dim + (size_t)Tprev * kv_dim, v, (size_t)T * kv_dim * sizeof(float));
    }

    /* q/k layernorm per head (gamma over head_dim) */
    for (int t = 0; t < T; t++) {
        for (int hh = 0; hh < nq; hh++)
            rmsnorm(q + (size_t)t * d + hh * hd, L->q_ln, hd, 1e-5f);
        for (int hh = 0; hh < nkv; hh++)
            rmsnorm(k + (size_t)t * kv_dim + hh * hd, L->k_ln, hd, 1e-5f);
    }

    /* RoPE on q,k (head_dim, theta) -- standard rotary, full dim */
    float theta = m->rope_theta;
    for (int t = 0; t < T; t++) {
        int pos = Tprev + t;
        for (int hh = 0; hh < nq; hh++) {
            float *qq = q + (size_t)t * d + hh * hd;
            for (int i = 0; i < hd; i += 2) {
                float freq = powf(theta, -2.0f * i / hd);
                float ang = pos * freq;
                float c = cosf(ang), s = sinf(ang);
                float a = qq[i], b = qq[i+1];
                qq[i] = a*c - b*s; qq[i+1] = a*s + b*c;
            }
        }
        for (int hh = 0; hh < nkv; hh++) {
            float *kk = k + (size_t)t * kv_dim + hh * hd;
            for (int i = 0; i < hd; i += 2) {
                float freq = powf(theta, -2.0f * i / hd);
                float ang = pos * freq;
                float c = cosf(ang), s = sinf(ang);
                float a = kk[i], b = kk[i+1];
                kk[i] = a*c - b*s; kk[i+1] = a*s + b*c;
            }
        }
    }

    /* attention: for each q head, average-pool over its kv head group (GQA) */
    float *out = (float *)xmalloc((size_t)T * d * sizeof(float));
    memset(out, 0, (size_t)T * d * sizeof(float));
    const float scale = 1.0f / sqrtf((float)hd);
    for (int t = 0; t < T; t++) {
        for (int hh = 0; hh < nq; hh++) {
            int kvh = hh / (nq / nkv);
            const float *Q = q + (size_t)t * d + hh * hd;
            float *O = out + (size_t)t * d + hh * hd;
            /* scores over all cached positions 0..Ttot-1 */
            float maxs = -1e30f;
            float *scores = (float *)xmalloc(Ttot * sizeof(float));
            for (int tp = 0; tp < Ttot; tp++) {
                const float *K = (kv_cache_layer ? kv_cache_layer + (size_t)tp * kv_dim + kvh * hd
                                                 : k + (size_t)tp * kv_dim + kvh * hd);
                float s = 0.0f;
                for (int i = 0; i < hd; i++) s += Q[i] * K[i];
                s *= scale;
                scores[tp] = s;
                if (s > maxs) maxs = s;
            }
            float sum = 0.0f;
            for (int tp = 0; tp < Ttot; tp++) { scores[tp] = expf(scores[tp] - maxs); sum += scores[tp]; }
            float inv = sum > 0 ? 1.0f/sum : 0.0f;
            for (int tp = 0; tp < Ttot; tp++) {
                const float *V = (kv_cache_layer ? kv_cache_layer + m->kv_max_t * kv_dim + (size_t)tp * kv_dim + kvh * hd
                                                 : v + (size_t)tp * kv_dim + kvh * hd);
                float wv = scores[tp] * inv;
                for (int i = 0; i < hd; i++) O[i] += wv * V[i];
            }
            free(scores);
        }
    }
    /* out_proj */
    matmul_f32(out, L->o_proj, T, d, d, attn_out);
    free(q); free(k); free(v); free(out);
}

/* Conv block forward (gated depthwise conv, no recurrence). */
static void lfm2_conv(const lfm2_layer_t *L, const lfm2_model_t *m,
                      const float *x, int T, float *out) {
    int d = m->d_model, cd = m->conv_dim, k = L->conv_k;
    float *proj = (float *)xmalloc((size_t)T * 3 * cd * sizeof(float));
    matmul_f32(x, L->in_proj, T, d, 3 * cd, proj);
    /* split B,C,h_tilde */
    const float *Bp = proj;
    const float *Cp = proj + (size_t)T * cd;
    const float *Hp = proj + (size_t)T * 2 * cd;
    float *y = (float *)xmalloc((size_t)T * cd * sizeof(float));
    for (size_t i = 0; i < (size_t)T * cd; i++) y[i] = Bp[i] * Hp[i];   /* input gate */
    float *z = (float *)xmalloc((size_t)T * cd * sizeof(float));
    depthwise_conv(y, L->conv_w, T, cd, k, z);                          /* conv */
    float *gated = (float *)xmalloc((size_t)T * cd * sizeof(float));
    for (size_t i = 0; i < (size_t)T * cd; i++) gated[i] = Cp[i] * z[i]; /* output gate */
    matmul_f32(gated, L->out_proj, T, cd, d, out);
    free(proj); free(y); free(z); free(gated);
}

/* SwiGLU FFN: h = w2( silu(w1(x)) * w3(x) ) */
static void lfm2_ffn(const lfm2_layer_t *L, const lfm2_model_t *m,
                     float *x, int T, float *out) {
    int d = m->d_model, ff = m->ff_dim;
    float *g = (float *)xmalloc((size_t)T * ff * sizeof(float));
    float *u = (float *)xmalloc((size_t)T * ff * sizeof(float));
    matmul_f32(x, L->w1, T, d, ff, g);
    matmul_f32(x, L->w3, T, d, ff, u);
    for (size_t i = 0; i < (size_t)T * ff; i++) {
        float v = g[i]; g[i] = v / (1.0f + expf(-v)) * u[i];  /* silu(g)*u */
    }
    matmul_f32(g, L->w2, T, ff, d, out);
    free(g); free(u);
}

/* ---- load ---- */
bool lfm2_load(const char *model_dir, lfm2_model_t *m) {
    memset(m, 0, sizeof(*m));
    /* Self-scan the directory for all model-NNN-of-MMM.safetensors shards and
     * open each (engine's wubu_shard_open scanner is unreliable here). */
    {
        DIR *d = opendir(model_dir);
        struct dirent *e;
        g_nsh = 0;
        if (d) {
            while ((e = readdir(d)) && g_nsh < 8) {
                const char *nm = e->d_name;
                if (strstr(nm, "model-") && strstr(nm, "-of-") && strstr(nm, ".safetensors")) {
                    char fp[2048]; snprintf(fp, sizeof(fp), "%s/%s", model_dir, nm);
                    st_ctx *s = st_open(fp);
                    fprintf(stderr, "[lfm2] shard %d: %s -> %s\n", g_nsh, fp, s ? "OK" : "FAIL");
                    if (s) g_shards[g_nsh++] = s;
                }
            }
            closedir(d);
        }
    }
    if (g_nsh == 0) { fprintf(stderr, "lfm2: no safetensors shards in %s\n", model_dir); return false; }
    fprintf(stderr, "[lfm2] opened %d shard(s)\n", g_nsh);

    /* dims from config -- passed via env or re-read; we read config.json */
    char cfg[2048]; snprintf(cfg, sizeof(cfg), "%s/config.json", model_dir);
    FILE *f = fopen(cfg, "rb");
    if (!f) { fprintf(stderr, "lfm2: no config.json\n"); return false; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f,0,SEEK_SET);
    char *buf = (char *)xmalloc(sz + 1); fread(buf, 1, sz, f); buf[sz]=0; fclose(f);
    m->d_model = 0; m->n_layers = 0; m->ff_dim = 0; m->n_q_heads = 0;
    m->n_kv_heads = 0; m->vocab_size = 0; m->conv_dim = 0;
    { const char *p = strstr(buf, "hidden_size"); if (p) { p = strchr(p, ':'); if (p) m->d_model = (int)atoll(p+1); } }
    { const char *p = strstr(buf, "num_hidden_layers"); if (p) { p = strchr(p, ':'); if (p) m->n_layers = (int)atoll(p+1); } }
    { const char *p = strstr(buf, "intermediate_size"); if (p) { p = strchr(p, ':'); if (p) m->ff_dim = (int)atoll(p+1); } }
    { const char *p = strstr(buf, "num_attention_heads"); if (p) { p = strchr(p, ':'); if (p) m->n_q_heads = (int)atoll(p+1); } }
    { const char *p = strstr(buf, "num_key_value_heads"); if (p) { p = strchr(p, ':'); if (p) m->n_kv_heads = (int)atoll(p+1); } }
    { const char *p = strstr(buf, "vocab_size"); if (p) { p = strchr(p, ':'); if (p) m->vocab_size = (int)atoll(p+1); } }
    { const char *p = strstr(buf, "conv_dim"); if (p) { p = strchr(p, ':'); if (p) m->conv_dim = (int)atoll(p+1); } }
    free(buf);
    m->head_dim = m->d_model / m->n_q_heads;
    m->rope_theta = 10000.0f; /* LFM2 default: config omits rope_theta -> 10000 */
    if (!m->d_model || !m->n_layers) { fprintf(stderr, "lfm2: bad config\n"); return false; }

    m->is_conv = (bool *)xmalloc(m->n_layers * sizeof(bool));
    m->layers = (lfm2_layer_t *)xmalloc(m->n_layers * sizeof(lfm2_layer_t));
    memset(m->layers, 0, m->n_layers * sizeof(lfm2_layer_t));

    /* layer_types: parse from config.json "layer_types" array (conv / full_attention) */
    FILE *fc = fopen(cfg, "rb"); fseek(fc,0,SEEK_END); long csz=ftell(fc); fseek(fc,0,SEEK_SET);
    char *cb=(char*)xmalloc(csz+1); fread(cb,1,csz,fc); cb[csz]=0; fclose(fc);
    const char *lt = strstr(cb, "\"layer_types\"");
    int li = 0;
    if (lt) { const char *p = strchr(lt, '['); if (p) { while(*++p && *p!=']' && li<m->n_layers) {
        if (*p=='"') { if (!strncmp(p+1,"conv",4)) m->is_conv[li++]=true;
                       else if (!strncmp(p+1,"full_attention",14)) m->is_conv[li++]=false;
                       while(*p&&*p!='"')p++; } } } }
    free(cb);
    if (li != m->n_layers) fprintf(stderr, "lfm2: warn layer_types count %d != %d\n", li, m->n_layers);

    /* per-layer weights */
    for (int l = 0; l < m->n_layers; l++) {
        lfm2_layer_t *L = &m->layers[l];
        L->conv_k = 3;
        char nm[128];
        #define LOAD(var, fmt) do { snprintf(nm,sizeof(nm),fmt,l); \
            L->var = load_bf16_f32(nm, NULL); \
            if (!L->var) { fprintf(stderr,"lfm2: missing %s\n", nm); } } while(0)
        if (m->is_conv[l]) {
            LOAD(in_proj, "model.layers.%d.conv.in_proj.weight");
            LOAD(conv_w,   "model.layers.%d.conv.conv.weight");
            LOAD(out_proj, "model.layers.%d.conv.out_proj.weight");
        } else {
            LOAD(q_proj, "model.layers.%d.self_attn.q_proj.weight");
            LOAD(k_proj, "model.layers.%d.self_attn.k_proj.weight");
            LOAD(v_proj, "model.layers.%d.self_attn.v_proj.weight");
            LOAD(o_proj, "model.layers.%d.self_attn.out_proj.weight");
            LOAD(q_ln,   "model.layers.%d.self_attn.q_layernorm.weight");
            LOAD(k_ln,   "model.layers.%d.self_attn.k_layernorm.weight");
        }
        LOAD(w1, "model.layers.%d.feed_forward.w1.weight");
        LOAD(w2, "model.layers.%d.feed_forward.w2.weight");
        LOAD(w3, "model.layers.%d.feed_forward.w3.weight");
        LOAD(ffn_norm, "model.layers.%d.ffn_norm.weight");
        LOAD(op_norm,  "model.layers.%d.operator_norm.weight");
        #undef LOAD
    }

    /* embeddings + norms (tied lm_head) */
    m->embed = load_bf16_f32("model.embed_tokens.weight", NULL);
    m->embed_norm = load_bf16_f32("model.embedding_norm.weight", NULL);
    m->kv_max_t = 8192;
    int kv_bytes = m->n_layers * 2 * m->n_kv_heads * m->head_dim * m->kv_max_t;
    m->kv_cache = (float *)xmalloc((size_t)kv_bytes * sizeof(float));
    memset(m->kv_cache, 0, (size_t)kv_bytes * sizeof(float));
    fprintf(stderr, "[lfm2] loaded d=%d layers=%d q=%d kv=%d hd=%d ff=%d vocab=%d conv_dim=%d\n",
            m->d_model, m->n_layers, m->n_q_heads, m->n_kv_heads, m->head_dim, m->ff_dim, m->vocab_size, m->conv_dim);
    return true;
}

void lfm2_free(lfm2_model_t *m) {
    if (m->layers) {
        for (int l = 0; l < m->n_layers; l++) {
            lfm2_layer_t *L = &m->layers[l];
            free(L->in_proj); free(L->conv_w); free(L->out_proj);
            free(L->q_proj); free(L->k_proj); free(L->v_proj); free(L->o_proj);
            free(L->q_ln); free(L->k_ln); free(L->w1); free(L->w2); free(L->w3);
            free(L->ffn_norm); free(L->op_norm);
        }
        free(m->layers);
    }
    free(m->is_conv); free(m->embed); free(m->embed_norm); free(m->kv_cache);
    for (int s = 0; s < g_nsh; s++) if (g_shards[s]) st_close(g_shards[s]);
    g_nsh = 0;
    memset(m, 0, sizeof(*m));
}

bool lfm2_forward(const lfm2_model_t *m, const float *emb, int B, int T, float *logits) {
    if (B != 1) { fprintf(stderr, "lfm2: only B=1 supported\n"); return false; }
    int d = m->d_model;
    float *h = (float *)xmalloc((size_t)T * d * sizeof(float));
    memcpy(h, emb, (size_t)T * d * sizeof(float));
    float *scratch = (float *)xmalloc((size_t)T * d * sizeof(float));
    float *tmp = (float *)xmalloc((size_t)T * d * sizeof(float));  /* normalized input */
    for (int l = 0; l < m->n_layers; l++) {
        const lfm2_layer_t *L = &m->layers[l];
        /* operator path: tmp = operator_norm(h); op writes to scratch;
         * residual = ORIGINAL h (not the normalized one). */
        for (int t = 0; t < T; t++) {
            memcpy(tmp + (size_t)t * d, h + (size_t)t * d, d * sizeof(float));
            rmsnorm(tmp + (size_t)t * d, L->op_norm, d, 1e-5f);
        }
        if (m->is_conv[l]) {
            lfm2_conv(L, m, tmp, T, scratch);
        } else {
            float *kvc = m->kv_cache + (size_t)l * 2 * m->n_kv_heads * m->head_dim * m->kv_max_t;
            lfm2_gqa(L, m, tmp, T, scratch, kvc);
        }
        /* residual add (h = h + op_result) */
        for (size_t i = 0; i < (size_t)T * d; i++) h[i] += scratch[i];
        /* ffn path: tmp = ffn_norm(h); ffn writes to scratch; residual add */
        for (int t = 0; t < T; t++) {
            memcpy(tmp + (size_t)t * d, h + (size_t)t * d, d * sizeof(float));
            rmsnorm(tmp + (size_t)t * d, L->ffn_norm, d, 1e-5f);
        }
        lfm2_ffn(L, m, tmp, T, scratch);
        for (size_t i = 0; i < (size_t)T * d; i++) h[i] += scratch[i];
        if (getenv("LFM2_DEBUG")) {
            const float *hp = h + (size_t)(T-1) * d;
            float ss = 0.0f; for (int q = 0; q < d; q++) ss += hp[q]*hp[q];
            fprintf(stderr, "L%d h_norm=%.4f", l, sqrtf(ss/d));
            if (l == 0) {
                const float *tp = tmp + (size_t)(T-1) * d;
                float ts = 0.0f; for (int q = 0; q < d; q++) ts += tp[q]*tp[q];
                const float *sp = scratch + (size_t)(T-1) * d;
                float sc = 0.0f; for (int q = 0; q < d; q++) sc += sp[q]*sp[q];
                fprintf(stderr, " tmp_norm=%.4f scratch_norm=%.4f", sqrtf(ts/d), sqrtf(sc/d));
            }
            fprintf(stderr, "\n");
        }
    }
    /* final embed norm + lm_head (t.; for next-token we need the last row. */
    rmsnorm(h + (size_t)(T-1) * d, m->embed_norm, d, 1e-5f);
    matmul_f32(h + (size_t)(T-1) * d, m->embed, 1, d, m->vocab_size, logits);
    free(h); free(scratch); free(tmp);
    return true;
}
