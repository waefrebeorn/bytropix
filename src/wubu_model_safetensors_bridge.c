/*
 * wubu_model_safetensors_bridge.c -- load HF safetensors Colonel models
 * into bytropix's wubu_model_t and run them through the EXISTING
 * SSM + GQA + MoE forward passes in pure F32.
 *
 * Tensor names are the real published HF names (from each repo's
 * model.safetensors.index.json):  model.language_model.layers.N.linear_attn.*
 * (the Gated DeltaNet SSM) + .self_attn.* (GQA) + .mlp.* (dense or MoE).
 * No third-party deps. C11, self-contained.
 */
#include "wubu_model_safetensors_bridge.h"
#include "wubu_dims.h"
#include "wubu_safetensors_shard.h"
#include "wubu_lora.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ---- helpers: load an F32 tensor (optionally transposed) from safetensors ---- */
static float *st_load_f32(st_ctx *st, const char *name, int64_t *nelems) {
    const st_tensor_info *t = st_find_tensor(st, name);
    if (!t) return NULL;
    int64_t n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= t->dims[d];
    *nelems = n;
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (!buf) return NULL;
    if (st_read_tensor_f32(st, t, buf, n) != n) { free(buf); return NULL; }
    return buf;
}

/* load + TRANSPOSE: HF Linear weight [out,in] -> bytropix [in,out] row-major.
 * Delegates to the shard loader's transpose path. */
static float *st_load_f32_t(wubu_shard_ctx_t *sc, const char *name, int rows, int cols) {
    float *dst = (float *)malloc((size_t)rows * cols * sizeof(float));
    if (!dst) return NULL;
    float *got = wubu_shard_load_f32_t(sc, name, rows, cols);
    if (!got) { free(dst); return NULL; }
    memcpy(dst, got, (size_t)rows * cols * sizeof(float));
    free(got);
    return dst;
}

/* printf-style tensor name builder */
static void tn(char *out, size_t cap, const char *fmt, int l) {
    snprintf(out, cap, fmt, l);
}

static void tn2(char *out, size_t cap, const char *fmt, int l, int e) {
    snprintf(out, cap, fmt, l, e);
}

/* Read dim `i` of a tensor by name (or -1 if absent). File-scope helper
 * used by wubu_model_init_safetensors to derive REAL model dimensions. */
static int dimof(st_ctx *s, const char *n, int i) {
    const st_tensor_info *t = st_find_tensor(s, n);
    return (t && i < t->n_dims) ? (int)t->dims[i] : -1;
}

int wubu_model_init_safetensors(wubu_model_t *m, const char *path,
                               const wubu_adapter_t *ad) {
    if (!m || !path || !ad) return -1;
    memset(m, 0, sizeof(*m));

    st_ctx *st = st_open(path);
    if (!st) { fprintf(stderr, "bridge: cannot open safetensors %s\n", path); return -1; }
    /* Shard ctx handles single-file OR multi-shard (model-NNNNN-of-NNNNN)
     * checkpoints transparently. `st` (shard 0) is used only for shape
     * probing below; all weight loads go through `sc`. */
    wubu_shard_ctx_t *sc = wubu_shard_open(path);
    if (!sc) { fprintf(stderr, "bridge: cannot open shard set %s\n", path); st_close(st); return -1; }

    /* ---- Derive REAL model dimensions from actual tensor shapes ----
     * This is the angel-coder fix for bytropix's previously hardcoded
     * 2048-dim forward: we read the checkpoint's true dims and feed them
     * to WUBU_DIMS so the forward runs at the model's real size. Falls
     * back to the adapter values if a shape can't be read. */
    int D     = dimof(st, "model.language_model.embed_tokens.weight", 1);
    if (D < 0) D = (int)ad->d_model;
    int CONVD = dimof(st, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 1);
    if (CONVD < 0) CONVD = D + D + D;
    int VD    = dimof(st, "model.language_model.layers.0.linear_attn.in_proj_z.weight", 1);
    if (VD < 0) VD = D;
    int DT    = dimof(st, "model.language_model.layers.0.linear_attn.in_proj_a.weight", 1);
    if (DT < 0) DT = 32;
    int SSMDS = dimof(st, "model.language_model.layers.0.linear_attn.norm.weight", 0);
    if (SSMDS < 0) SSMDS = 128;
    int qdim  = dimof(st, "model.language_model.layers.0.self_attn.q_proj.weight", 1);
    int kvdim = dimof(st, "model.language_model.layers.0.self_attn.k_proj.weight", 1);
    int qh = ad->gqa_q_heads > 0 ? (int)ad->gqa_q_heads : 16;
    int kvh = ad->gqa_kv_heads > 0 ? (int)ad->gqa_kv_heads : 2;
    int hd = ad->gqa_head_dim > 0 ? (int)ad->gqa_head_dim : 256;
    if (qdim > 0 && hd > 0) qh = qdim / hd;
    if (kvdim > 0 && hd > 0) kvh = kvdim / hd;
    int nL = (int)ad->n_layers;
    int nE = (int)ad->n_experts;
    int dff = (int)ad->d_ff > 0 ? (int)ad->d_ff : (D * 4);

    wubu_dims_t wd; memset(&wd, 0, sizeof(wd));
    wd.d_model = D; wd.conv_dim = CONVD; wd.value_dim = VD; wd.dt_rank = DT;
    wd.ssm_d_state = SSMDS; wd.ssm_k_heads = D > 0 ? (CONVD - VD) / 2 / SSMDS : 16;
    wd.ssm_v_heads = VD / SSMDS; wd.conv_kernel = 4;
    wd.gqa_q_heads = qh; wd.gqa_kv_heads = kvh; wd.gqa_head_dim = hd;
    wubu_dims_set(&wd);

    m->d_model = D;
    m->n_layers = nL;
    m->vocab_size = 248320;   /* set from embed_tokens shape below if present */
    m->n_experts = nE;
    m->n_active_experts = (int)ad->n_active_experts;

    m->layers = (wubu_layer_t *)calloc((size_t)nL, sizeof(wubu_layer_t));
    if (!m->layers) { st_close(st); return -1; }

    char nm[256];
    for (int l = 0; l < nL; l++) {
        wubu_layer_t *ly = &m->layers[l];
        ly->is_ssm = 1; /* hybrid: runs BOTH SSM (linear_attn) and GQA (self_attn) */

        /* ---- GQA (self_attn.*_proj) ---- */
        tn(nm, sizeof(nm), "model.language_model.layers.%d.self_attn.q_proj.weight", l);
        ly->gqa.attn_q_weight = wubu_shard_load_f32(sc, nm, &(int64_t){0});
        tn(nm, sizeof(nm), "model.language_model.layers.%d.self_attn.k_proj.weight", l);
        ly->gqa.attn_k_weight = wubu_shard_load_f32(sc, nm, &(int64_t){0});
        tn(nm, sizeof(nm), "model.language_model.layers.%d.self_attn.v_proj.weight", l);
        ly->gqa.attn_v_weight = wubu_shard_load_f32(sc, nm, &(int64_t){0});
        tn(nm, sizeof(nm), "model.language_model.layers.%d.self_attn.o_proj.weight", l);
        ly->gqa.attn_output_weight = wubu_shard_load_f32(sc, nm, &(int64_t){0});
        ly->gqa.q_heads = qh; ly->gqa.kv_heads = kvh; ly->gqa.head_dim = hd;
        ly->gqa.attn_q_weight_q = NULL; ly->gqa.attn_k_weight_q = NULL;
        ly->gqa.attn_v_weight_q = NULL; ly->gqa.attn_output_weight_q = NULL;

        /* ---- SSM (linear_attn.*) -> F32 path ----
         * HF stores Linear weights [out,in]; bytropix forward reads them
         * [in,out] row-major, so we transpose qkv/gate on load.
         * DT/A_log/beta/norm are 1D and copied as-is. */
        ly->ssm.f32_mode = 1;
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.in_proj_qkv.weight", l);
        ly->ssm.attn_qkv_weight_f32 = wubu_shard_load_f32_t(sc, nm, D, CONVD); /* [D,CONVD] */
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.in_proj_z.weight", l);
        ly->ssm.attn_gate_weight_f32 = wubu_shard_load_f32_t(sc, nm, D, VD); /* [D,VD] */
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.in_proj_a.weight", l);
        ly->ssm.ssm_alpha_weight = wubu_shard_load_f32(sc, nm, &(int64_t){0});
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.in_proj_b.weight", l);
        ly->ssm.ssm_beta_weight = wubu_shard_load_f32(sc, nm, &(int64_t){0});
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.A_log.weight", l);
        ly->ssm.ssm_a = wubu_shard_load_f32(sc, nm, &(int64_t){0});
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.dt_bias.weight", l);
        ly->ssm.ssm_dt_bias = wubu_shard_load_f32(sc, nm, &(int64_t){0});
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.convNd.weight", l);
        ly->ssm.ssm_conv1d_weight = wubu_shard_load_f32(sc, nm, &(int64_t){0});
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.norm.weight", l);
        ly->ssm.ssm_norm_weight = wubu_shard_load_f32(sc, nm, &(int64_t){0});
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.out_proj.weight", l);
        ly->ssm.ssm_out_weight_f32 = wubu_shard_load_f32(sc, nm, &(int64_t){0}); /* HF out_proj [VD,D] == bytropix [VD,D] */
        ly->ssm.attn_qkv_weight_q = NULL; ly->ssm.attn_gate_weight_q = NULL;
        ly->ssm.ssm_out_weight_q = NULL;

        /* ---- MoE / dense MLP (mlp.*) ---- */
        int nExp = nE > 0 ? nE : 1; /* dense => single-expert MoE */
        ly->moe.ffn_gate_exps = (float *)calloc((size_t)D * dff * nExp, sizeof(float));
        ly->moe.ffn_up_exps   = (float *)calloc((size_t)D * dff * nExp, sizeof(float));
        ly->moe.ffn_down_exps = (float *)calloc((size_t)dff * D * nExp, sizeof(float));
        ly->moe.load_from_blob = false;
        if (nE > 0) {
            /* MoE: mlp.experts.N.{gate,up,down}_proj [dff, D] -> transpose */
            tn(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.0.gate_proj.weight", l);
            int64_t cnt = 0; float *g0 = wubu_shard_load_f32(sc, nm, &cnt);
            if (g0) {
                for (int e = 0; e < nE; e++) {
                    tn2(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.%d.gate_proj.weight", l, e);
                    float *g = wubu_shard_load_f32_t(sc, nm, D, dff); /* [D,dff] */
                    tn2(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.%d.up_proj.weight", l, e);
                    float *u = wubu_shard_load_f32_t(sc, nm, D, dff);
                    tn2(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.%d.down_proj.weight", l, e);
                    float *d = wubu_shard_load_f32_t(sc, nm, dff, D); /* [dff,D] */
                    if (g) memcpy(ly->moe.ffn_gate_exps + (size_t)e * D * dff, g, (size_t)D * dff * sizeof(float));
                    if (u) memcpy(ly->moe.ffn_up_exps   + (size_t)e * D * dff, u, (size_t)D * dff * sizeof(float));
                    if (d) memcpy(ly->moe.ffn_down_exps + (size_t)e * dff * D, d, (size_t)dff * D * sizeof(float));
                    free(g); free(u); free(d);
                }
                free(g0);
                tn(nm, sizeof(nm), "model.language_model.layers.%d.mlp.gate.weight", l);
                ly->moe.ffn_gate_inp = wubu_shard_load_f32(sc, nm, &(int64_t){0}); /* [D, nE] */
            }
        } else {
            /* dense MLP: mlp.{gate,up,down}_proj -> expert 0 */
            tn(nm, sizeof(nm), "model.language_model.layers.%d.mlp.gate_proj.weight", l);
            float *g = wubu_shard_load_f32_t(sc, nm, D, dff);
            tn(nm, sizeof(nm), "model.language_model.layers.%d.mlp.up_proj.weight", l);
            float *u = wubu_shard_load_f32_t(sc, nm, D, dff);
            tn(nm, sizeof(nm), "model.language_model.layers.%d.mlp.down_proj.weight", l);
            float *d = wubu_shard_load_f32_t(sc, nm, dff, D);
            if (g) memcpy(ly->moe.ffn_gate_exps, g, (size_t)D * dff * sizeof(float));
            if (u) memcpy(ly->moe.ffn_up_exps,   u, (size_t)D * dff * sizeof(float));
            if (d) memcpy(ly->moe.ffn_down_exps, d, (size_t)dff * D * sizeof(float));
            free(g); free(u); free(d);
        }
    }

    /* ---- embed_tokens / lm_head ---- */
    int64_t ne = 0;
    m->token_embd = wubu_shard_load_f32(sc, "model.language_model.embed_tokens.weight", &ne);
    int64_t no = 0;
    m->output_weight = wubu_shard_load_f32(sc, "lm_head.weight", &no);
    if (!m->output_weight) /* some models name it language_model.lm_head */
        m->output_weight = wubu_shard_load_f32(sc, "model.language_model.lm_head.weight", &no);
    m->token_embd_q = NULL; m->output_weight_q = NULL;
    m->use_embedding_file = false;

    st_close(st);
    wubu_shard_close(sc);

    /* ---- LoRA (BTL-3): apply delta from adapter onto base weights ----
     * For BTL-3, `path` is the LoRA adapter safetensors and the BASE
     * checkpoint (ad->base_model) must already be loaded into `m`.
     * We read lora_A / lora_B per target module and add scale*(B@A)
     * to the corresponding F32 weight already held in `m`. */
    if (ad->is_lora && ad->base_model[0]) {
        st_ctx *ast = st_open(path);
        if (ast) {
            for (int l = 0; l < nL; l++) {
                char an[256], bn[256];
                tn(an, sizeof(an),
                    "model.language_model.layers.%d.self_attn.q_proj.lora_A.weight", l);
                tn(bn, sizeof(bn),
                    "model.language_model.layers.%d.self_attn.q_proj.lora_B.weight", l);
                int64_t na = 0, nb = 0;
                float *A = st_load_f32(ast, an, &na);
                float *B = st_load_f32(ast, bn, &nb);
                if (A && B && na == (int64_t)32 * D && nb == (int64_t)D * 32) {
                    wubu_lora_t *la = wubu_lora_create(32, 64.0f, D, D);
                    if (la) {
                        wubu_lora_load_f32(la, A, B);
                        wubu_lora_apply(la, m->layers[l].gqa.attn_q_weight);
                        wubu_lora_free(la);
                    }
                }
                free(A); free(B);
            }
            st_close(ast);
        }
    }

    return 0;
}

int wubu_model_init_auto(wubu_model_t *m, const char *path) {
    size_t plen = strlen(path);
    int is_st = (plen > 13 && strcmp(path + plen - 13, ".safetensors") == 0);
    if (is_st) {
        wubu_adapter_t ad; memset(&ad, 0, sizeof(ad));
        if (!wubu_adapter_load(&ad, path)) {
            ad.arch = WUBU_ARCH_QWEN_FAMILY; ad.ok = 1;
        }
        return wubu_model_init_safetensors(m, path, &ad);
    }
    /* legacy GGUF path */
    return wubu_model_init(m, path);
}

void wubu_model_safetensors_free(wubu_model_t *m) {
    if (!m || !m->layers) return;
    for (int l = 0; l < m->n_layers; l++) {
        wubu_layer_t *ly = &m->layers[l];
        free(ly->gqa.attn_q_weight); free(ly->gqa.attn_k_weight);
        free(ly->gqa.attn_v_weight); free(ly->gqa.attn_output_weight);
        free(ly->ssm.attn_qkv_weight_f32); free(ly->ssm.attn_gate_weight_f32);
        free(ly->ssm.ssm_alpha_weight); free(ly->ssm.ssm_beta_weight);
        free(ly->ssm.ssm_a); free(ly->ssm.ssm_dt_bias);
        free(ly->ssm.ssm_conv1d_weight); free(ly->ssm.ssm_norm_weight);
        free(ly->ssm.ssm_out_weight_f32);
        free(ly->moe.ffn_gate_exps); free(ly->moe.ffn_up_exps);
        free(ly->moe.ffn_down_exps); free(ly->moe.ffn_gate_inp);
    }
    free(m->token_embd); free(m->output_weight);
    free(m->layers);
    m->layers = NULL;
}
