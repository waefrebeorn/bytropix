/*
 * wubu_model_adapter.c -- HF config.json -> bytropix dims (C11,
 * self-contained, opaque). Hand-parses the JSON we care about:
 *   architectures[0] / model_type, num_hidden_layers, hidden_size,
 *   intermediate_size, num_experts, num_experts_per_tok, num_attention_heads,
 *   num_key_value_heads, head_dim, rope_theta, partial_rotary_factor,
 *   vocab_size, and (BTL-3) base_model / adapter info.
 *
 * No external JSON lib required.
 */

#include "wubu_model_adapter.h"
#include <stdlib.h>
#include <string.h>
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

static void apply_qwen36_defaults(wubu_adapter_t *a) {
    // Qwen3.6-35B-A3B derived defaults (bytropix macros).
    a->tensor_naming = 0;          // blk.Qwen naming for GGUF; but HF uses model.layers.*
    a->d_model = a->d_model ? a->d_model : 2048;
    a->gqa_q_heads = a->gqa_q_heads ? a->gqa_q_heads : 16;
    a->gqa_kv_heads = a->gqa_kv_heads ? a->gqa_kv_heads : 2;
    a->gqa_head_dim = a->gqa_head_dim ? a->gqa_head_dim : 256;
    a->rope_theta = a->rope_theta ? a->rope_theta : 10000000.0f;
    a->partial_rotary_factor = a->partial_rotary_factor ? a->partial_rotary_factor : 0.25f;
    a->ssm_v_heads = 32;
    a->ssm_d_state = 128;
    a->d_ff = a->d_ff ? a->d_ff : 512;   // MoE expert dim default
}

bool wubu_adapter_load(wubu_adapter_t *out, const char *path) {
    if (!out || !path) return false;
    memset(out, 0, sizeof(*out));
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
    out->n_experts = (int)find_int(buf, end, "num_experts", 0);
    out->n_active_experts = (int)find_int(buf, end, "num_experts_per_tok", 0);
    out->gqa_q_heads = (int)find_int(buf, end, "num_attention_heads", 0);
    out->gqa_kv_heads = (int)find_int(buf, end, "num_key_value_heads", 0);
    out->gqa_head_dim = (int)find_int(buf, end, "head_dim", 0);
    out->rope_theta = find_float(buf, end, "rope_theta", 0.0f);
    out->partial_rotary_factor = find_float(buf, end, "partial_rotary_factor", 0.0f);

    const char *arch = find_str(buf, end, "architectures");
    const char *mt = find_str(buf, end, "model_type");
    const char *base = find_str(buf, end, "base_model");

    // Detect LoRA adapter (BTL-3 style: has base_model + adapter fields)
    if (base) {
        int bi = 0;
        while (base[bi] && base[bi] != '"' && bi < (int)sizeof(out->base_model) - 1) {
            out->base_model[bi] = base[bi]; bi++;
        }
        out->base_model[bi] = '\0';
        out->is_lora = true;
        out->arch = WUBU_ARCH_BTL3_LORA;
        apply_qwen36_defaults(out);
        out->ok = true;
        free(buf);
        return true;
    }

    // Detect MoE
    if (out->n_experts > 0) {
        out->is_moe = true;
        out->arch = WUBU_ARCH_KAT_MOE;
        apply_qwen36_defaults(out);
        out->ok = true;
        free(buf);
        return true;
    }

    // Dense Qwen-family (Qwen3.6 / Agents-A1)
    out->arch = WUBU_ARCH_QWEN_FAMILY;
    apply_qwen36_defaults(out);
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
        out->d_model = 2048;       // Qwen3.6-35B-A3B hidden
        out->n_experts = 35;        // spec: 35B/3B activated (MoE)
        out->n_active_experts = 3;
        out->n_layers = 64;
        out->gqa_head_dim = 256;
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
     * bytropix already supports; the on-disk gguf carries the real dims. */
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
        default:                      return "unknown";
    }
}
