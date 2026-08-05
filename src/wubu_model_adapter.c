/*
 * wubu_model_adapter.c -- HF config.json -> wubuwizard dims (C11,
 * self-contained, opaque). Hand-parses the JSON we care about:
 *   architectures[0] / model_type, num_hidden_layers, hidden_size,
 *   intermediate_size, num_experts, num_experts_per_tok, num_attention_heads,
 *   num_key_value_heads, head_dim, rope_theta, partial_rotary_factor,
 *   vocab_size, and (BTL-3) base_model / adapter info.
 *
 * No external JSON lib required.
 */

#include "wubu_model_adapter.h"
#include "wubu_da_guard.h"
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <stdio.h>

/* ---- minimal JSON scalar reader (matches "key" : value) ---- */
static const char *find_str(const char *p, const char *end, const char *key) {
    size_t klen = strlen(key);
    for (; p + klen + 2 < end; p++) {
        if (p[0] == '"' && strncmp(p + 1, key, klen) == 0 && p[1 + klen] == '"') {
            const char *q = p + 1 + klen + 1;
            while (q < end && *q != ':') q++;
            if (q >= end) return NULL;
            q++;
            while (q < end && (*q == ' ' || *q == '\t' || *q == '\n')) q++;
            if (q < end && *q == '"') return q + 1;
            return NULL;
        }
    }
    return NULL;
}

static int64_t read_int(const char **pp, const char *end) {
    const char *p = *pp;
    while (p < end && (*p < '0' || *p > '9') && *p != '-') p++;
    if (p >= end) { *pp = p; return 0; }
    int64_t sign = 1, val = 0;
    if (*p == '-') { sign = -1; p++; }
    while (p < end && *p >= '0' && *p <= '9') { val = val * 10 + (*p - '0'); p++; }
    *pp = p;
    return val * sign;
}

static int64_t find_int(const char *p, const char *end, const char *key, int64_t def) {
    size_t klen = strlen(key);
    for (; p + klen + 2 < end; p++) {
        if (p[0] == '"' && strncmp(p + 1, key, klen) == 0 && p[1 + klen] == '"') {
            const char *q = p + 1 + klen + 1;  /* past closing quote */
            while (q < end && *q != ':') q++;
            if (q >= end) return def;
            q++;
            while (q < end && (*q == ' ' || *q == '\t' || *q == '\n')) q++;
            const char *r = q;
            return read_int(&r, end);
        }
    }
    return def;
}

static float find_float(const char *p, const char *end, const char *key, float def) {
    int64_t i = find_int(p, end, key, (int64_t)def);
    return (float)i;
}

/* Parse the layer_types array (list of "linear_attention"/"full_attention"
 * strings) into out->layer_types[]. Returns count, or 0 if absent. */
static int parse_layer_types(wubu_adapter_t *out, const char *buf, const char *end) {
    const char *p = strstr(buf, "\"layer_types\"");
    if (!p) return 0;
    p = strchr(p, '[');
    if (!p) return 0;
    int n = 0;
    while (++p < end && *p != ']' && n < 256) {
        if (*p == '"') {
            if (strncmp(p + 1, "full_attention", 14) == 0) out->layer_types[n++] = 1;
            else if (strncmp(p + 1, "linear_attention", 15) == 0) out->layer_types[n++] = 0;
            while (p < end && *p != '"') p++;
        }
    }
    return n;
}

bool wubu_adapter_load(wubu_adapter_t *out, const char *path) {
    if (!out || !path) return false;
    /* DA-2 fail-closed: refuse to load weights if the kernel schema
     * doesn't match our compile-time constant. The env var WUBU_KERNEL_SCHEMA
     * is set by wubu_realm_start() in WuBuOS; absent it, we run standalone. */
    if (wubu_da_check_kernel_schema() != 0) return false;
    memset(out, 0, sizeof(*out));
    /* A bare checkpoint DIRECTORY: resolve to its config.json. The model dir
     * (model-NNN-of-MMM shards) carries config.json; opening the dir directly
     * with fopen() succeeds on Linux but ftell() returns garbage -> giant alloc. */
    char cfg_path[2048];
    cfg_path[0] = 0;
    struct stat pst;
    if (stat(path, &pst) == 0 && S_ISDIR(pst.st_mode)) {
        snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", path);
        path = cfg_path;
    }
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return false; }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return false; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return false; }
    fclose(f);
    const char *end = buf + sz;

    out->d_model = (int)find_int(buf, end, "hidden_size", 0);
    out->n_layers = (int)find_int(buf, end, "num_hidden_layers", 0);
    out->d_ff = (int)find_int(buf, end, "intermediate_size", 0);
    out->d_ff = out->d_ff ? out->d_ff : (int)find_int(buf, end, "moe_intermediate_size", 0);
    out->n_experts = (int)find_int(buf, end, "num_experts", 0);
    out->n_active_experts = (int)find_int(buf, end, "num_experts_per_tok", 0);
    out->gqa_q_heads = (int)find_int(buf, end, "num_attention_heads", 0);
    out->gqa_kv_heads = (int)find_int(buf, end, "num_key_value_heads", 0);
    out->gqa_head_dim = (int)find_int(buf, end, "head_dim", 0);
    out->rope_theta = find_float(buf, end, "rope_theta", 0.0f);
    out->partial_rotary_factor = find_float(buf, end, "partial_rotary_factor", 0.0f);
    out->ssm_k_heads = (int)find_int(buf, end, "linear_num_key_heads", 0);
    out->ssm_v_heads = (int)find_int(buf, end, "linear_num_value_heads", 0);
    out->ssm_value_head_dim = (int)find_int(buf, end, "linear_value_head_dim", 0);
    out->ssm_conv_kernel = (int)find_int(buf, end, "linear_conv_kernel_dim", 0);
    out->ssm_d_state = (int)find_int(buf, end, "ssm_d_state", 0);
    if (!out->ssm_d_state) out->ssm_d_state = (int)find_int(buf, end, "state_size", 128);
    out->shared_expert_ff = (int)find_int(buf, end, "shared_expert_intermediate_size", 0);
    out->full_attention_interval = (int)find_int(buf, end, "full_attention_interval", 0);
    out->attn_output_gate = find_int(buf, end, "attn_output_gate", 0) != 0;
    out->vocab_size = (int)find_int(buf, end, "vocab_size", 0);

    /* Some HF configs (Qwen3.6, Agents-A1, KAT-Coder) nest all the real
     * architecture fields inside a "text_config" object. If the top-level
     * scan yielded nothing, re-scan just the text_config {...} span. */
    if (out->d_model == 0) {
        const char *tc = strstr(buf, "\"text_config\"");
        if (tc && tc < end) {
            const char *open = strchr(tc, '{');
            if (open) {
                int depth = 0; const char *e2 = open;
                for (; e2 < end; e2++) {
                    if (*e2 == '{') depth++;
                    else if (*e2 == '}') { depth--; if (depth == 0) break; }
                }
                const char *tc_end = (e2 < end) ? e2 + 1 : end;
                out->d_model = (int)find_int(open, tc_end, "hidden_size", 0);
                out->n_layers = (int)find_int(open, tc_end, "num_hidden_layers", 0);
                out->d_ff = (int)find_int(open, tc_end, "intermediate_size", 0);
                out->d_ff = out->d_ff ? out->d_ff : (int)find_int(open, tc_end, "moe_intermediate_size", 0);
                out->n_experts = (int)find_int(open, tc_end, "num_experts", 0);
                out->n_active_experts = (int)find_int(open, tc_end, "num_experts_per_tok", 0);
                out->gqa_q_heads = (int)find_int(open, tc_end, "num_attention_heads", 0);
                out->gqa_kv_heads = (int)find_int(open, tc_end, "num_key_value_heads", 0);
                out->gqa_head_dim = (int)find_int(open, tc_end, "head_dim", 0);
                out->rope_theta = find_float(open, tc_end, "rope_theta", 0.0f);
                out->partial_rotary_factor = find_float(open, tc_end, "partial_rotary_factor", 0.0f);
                out->ssm_k_heads = (int)find_int(open, tc_end, "linear_num_key_heads", 0);
                out->ssm_v_heads = (int)find_int(open, tc_end, "linear_num_value_heads", 0);
                out->ssm_value_head_dim = (int)find_int(open, tc_end, "linear_value_head_dim", 0);
                out->ssm_conv_kernel = (int)find_int(open, tc_end, "linear_conv_kernel_dim", 0);
                out->ssm_d_state = (int)find_int(open, tc_end, "ssm_d_state", 0);
                if (!out->ssm_d_state) out->ssm_d_state = (int)find_int(open, tc_end, "state_size", 128);
                out->shared_expert_ff = (int)find_int(open, tc_end, "shared_expert_intermediate_size", 0);
                out->full_attention_interval = (int)find_int(open, tc_end, "full_attention_interval", 0);
                out->attn_output_gate = find_int(open, tc_end, "attn_output_gate", 0) != 0;
                out->vocab_size = (int)find_int(open, tc_end, "vocab_size", 0);
                int nlt = parse_layer_types(out, open, tc_end);
                if (nlt > 0) out->is_hybrid = true;
            }
        }
    }

    /* Always try to parse hybrid layer_types (top-level or inside text_config). */
    {
        int nlt = parse_layer_types(out, buf, end);
        if (nlt > 0) out->is_hybrid = true;
    }

    const char *arch = find_str(buf, end, "architectures");
    const char *mt = find_str(buf, end, "model_type");
    /* base_model / base_model_name_or_path is a SUBSTRING of the metadata
     * key. A plain strstr() won't work here: buf is not null-terminated
     * and the 8-byte safetensors length prefix contains embedded NULs
     * (e.g. 2073 -> 0x17 0x08 0x00 ...) that terminate strstr at
     * the prefix before the JSON header is ever reached. Use a
     * length-bounded search instead. */
    const char *base = memmem(buf, (size_t)(end - buf), "base_model", 11);
    if (!base) base = memmem(buf, (size_t)(end - buf), "base_model_name_or_path", 22);
    if (base) {
        const char *q = base;
        while (q < end && *q != ':') q++;
        if (q < end) {
            q++;
            while (q < end && (*q == ' ' || *q == '\t' || *q == '\n')) q++;
            if (q < end && *q == '"') {
                const char *vs = q + 1;
                int bi = 0;
                while (vs[bi] && vs[bi] != '"' && bi < (int)sizeof(out->base_model) - 1) {
                    out->base_model[bi] = vs[bi]; bi++;
                }
                out->base_model[bi] = '\0';
            }
        }
    }

    // Detect LoRA adapter (BTL-3 style: has base_model + adapter fields)
    if (base) {
        out->is_lora = true;
        out->lora_r = 32;        // BTL-3 LoRA rank
        out->lora_alpha = 64;     // BTL-3 LoRA alpha
        out->arch = WUBU_ARCH_BTL3_LORA;
        out->ok = true;
        free(buf);
        return true;
    }

    // Detect MoE — check for DeepSeek-V4-Flash first (MXFP4 experts)
    if (out->n_experts > 0) {
        out->is_moe = true;
        if (mt && memmem(buf, (size_t)(end - buf), "deepseek_v4", 11)) {
            out->arch = WUBU_ARCH_DEEPSEEK_V4_MOE;
            /* DeepSeek V4 Flash uses shared_experts + routed experts.
             * Shared experts are dense (FP8/BF16), routed are MXFP4.
             * The GGUF quant_type field tells us at load time which is which. */
            out->shared_expert_ff = out->shared_expert_ff
                ? out->shared_expert_ff
                : out->d_ff;
            /* DeepSeek V4 MLA attention: Q uses latent dim, KV uses grouped-decomp */
            out->is_hybrid = false;  /* MLA is not SSM+GQA hybrid */
        } else {
            out->arch = WUBU_ARCH_KAT_MOE;
        }
        out->ok = true;
        free(buf);
        return true;
    }

    // Dense Qwen-family (Qwen3.6 / Agents-A1)
    out->arch = WUBU_ARCH_QWEN_FAMILY;
    out->ok = true;
    free(buf);
    return true;
}

bool wubu_adapter_resolve_name(wubu_adapter_t *out, const char *name) {
    if (!out || !name) return false;
    memset(out, 0, sizeof(*out));
    const char *n = name;
    if (strstr(n, "BTL-3") || strstr(n, "btl3")) {
        out->arch = WUBU_ARCH_BTL3_LORA;
        strncpy(out->base_model, "Qwen/Qwen3.6-27B", sizeof(out->base_model) - 1);
        out->is_lora = true;
        out->lora_r = 32;        // BTL-3 LoRA rank
        out->lora_alpha = 64;     // BTL-3 LoRA alpha
        out->d_model = 5120;       // Qwen3.6-27B hidden
        out->n_layers = 64;
        out->gqa_head_dim = 256;
        out->gqa_kv_heads = 4;       // Qwen3.6-27B KV heads
        out->gqa_q_heads = 24;
        out->ssm_v_heads = 48;       // Qwen3.6-27B ssm_v
        out->ssm_d_state = 128;
        out->ok = true;
        return true;
    }
    if (strstr(n, "KAT-Coder")) {
        out->arch = WUBU_ARCH_KAT_MOE;
        out->is_moe = true;
        out->d_model = 2048;
        out->n_experts = 256;        // real KAT-Coder: 256 routed experts
        out->n_active_experts = 8;   // num_experts_per_tok
        out->n_layers = 40;          // num_hidden_layers
        out->d_ff = 512;             // moe_intermediate_size
        out->gqa_q_heads = 16;
        out->gqa_kv_heads = 2;
        out->gqa_head_dim = 256;
        out->ssm_k_heads = 16;
        out->ssm_v_heads = 32;
        out->ssm_value_head_dim = 128;
        out->ssm_conv_kernel = 4;
        out->ssm_d_state = 128;
        out->shared_expert_ff = 512;
        out->full_attention_interval = 4;
        out->attn_output_gate = true;
        out->partial_rotary_factor = 0.25f;
        out->ok = true;
        return true;
    }
    if (strstr(n, "Agents-A1")) {
        out->arch = WUBU_ARCH_QWEN_FAMILY;
        out->d_model = 5120;       // ~5B dense Qwen-family
        out->n_layers = 40;        // heuristic
        out->gqa_head_dim = 256;
        out->ok = true;
        return true;
    }
    if (strstr(n, "Qwen3.6-27B")) {
        out->arch = WUBU_ARCH_QWEN36_HYBRID;
        out->d_model = 5120;
        out->n_layers = 64;
        out->gqa_head_dim = 256;
        out->gqa_kv_heads = 4;
        out->gqa_q_heads = 24;
        out->ssm_v_heads = 48;
        out->ssm_d_state = 128;
        out->ok = true;
        return true;
    }
    /* DeepSeek-V4-Flash: 284B MoE, MXFP4 experts, MLA attention, 1M ctx */
    if (strstr(n, "DeepSeek-V4") || strstr(n, "deepseek") || strstr(n, "ds4")) {
        out->arch = WUBU_ARCH_DEEPSEEK_V4_MOE;
        out->is_moe = true;
        out->n_experts = 256;
        out->n_active_experts = 6;
        out->n_layers = 43;
        out->d_model = 7168;
        out->d_ff = 18432;
        out->shared_expert_ff = 18432;
        out->gqa_q_heads = 128;
        out->gqa_kv_heads = 8;
        out->gqa_head_dim = 128;
        out->rope_theta = 1e8f;  /* 100M rope_theta for MLA */
        out->vocab_size = 129280;
        out->partial_rotary_factor = 0.5f;
        out->ok = true;
        return true;
    }
    return false;
}

/* Resolve a model checkpoint path to its Colonel identity.
 * Handles both legacy .gguf and the new .safetensors Colonel models. */
bool wubu_model_resolve(const char *path, wubu_adapter_t *out) {
    if (!path || !out) return false;
    memset(out, 0, sizeof(*out));
    size_t plen = strlen(path);
    int is_safetensors = (plen > 13 && strcmp(path + plen - 13, ".safetensors") == 0);

    if (is_safetensors) {
        if (!wubu_adapter_load(out, path)) {
            out->arch = WUBU_ARCH_QWEN_FAMILY;
            out->ok = true;
            return true;
        }
        return true;
    }

    /* Legacy GGUF: report the generic Qwen3.6-35B-A3B family that
     * wubuwizard already supports; the on-disk gguf carries the real dims. */
    out->arch = WUBU_ARCH_QWEN_FAMILY;
    out->tensor_naming = 0;
    out->ok = true;
    return true;
}

const char *wubu_arch_name(wubu_arch_t a) {
    switch (a) {
        case WUBU_ARCH_QWEN36_HYBRID: return "Qwen3.6-27B hybrid";
        case WUBU_ARCH_QWEN_FAMILY:   return "Qwen-family dense";
        case WUBU_ARCH_KAT_MOE:       return "KAT-Coder MoE";
        case WUBU_ARCH_BTL3_LORA:    return "BTL-3 LoRA";
        case WUBU_ARCH_DEEPSEEK_V4_MOE: return "DeepSeek-V4-Flash MoE";
        default:                      return "unknown";
    }
}
