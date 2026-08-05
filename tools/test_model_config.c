/*
 * test_model_config.c -- verify the HF adapter derives the REAL model dims
 * from config.json for the four target models. Memory-safe: reads only the
 * small config.json, never opens the weights.
 */
#include "wubu_model_adapter.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>

/* Models live on persistent storage (we lost /tmp weights to a reboot/cleanup).
 * WUBUWIZARD_MODELS_DIR overrides; default points at the persistent vault. */
static const char *models_dir(void) {
    const char *d = getenv("WUBUWIZARD_MODELS_DIR");
    return d && *d ? d : "/home/wubu/models";
}

static int exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static int check(const char *name, int got, int want) {
    int ok = (got == want);
    printf("  %-22s got=%-6d want=%-6d %s\n", name, got, want, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

static int test_one(const char *path, const char *label,
                    int d_model, int n_layers, int n_experts, int n_active,
                    int ssm_k, int ssm_v, int ssm_vdim, int conv, int shared,
                    int fa_int, int is_moe, int is_hybrid) {
    printf("=== %s ===\n", label);
    wubu_adapter_t a; memset(&a, 0, sizeof(a));
    if (!wubu_adapter_load(&a, path)) { printf("  FAIL: cannot load %s\n", path); return 1; }
    int f = 0;
    f += check("d_model",        a.d_model,        d_model);
    f += check("n_layers",       a.n_layers,       n_layers);
    f += check("n_experts",      a.n_experts,      n_experts);
    f += check("n_active",       a.n_active_experts, n_active);
    f += check("ssm_k_heads",    a.ssm_k_heads,    ssm_k);
    f += check("ssm_v_heads",    a.ssm_v_heads,    ssm_v);
    f += check("ssm_vdim",       a.ssm_value_head_dim, ssm_vdim);
    f += check("conv_kernel",    a.ssm_conv_kernel, conv);
    f += check("shared_expert",  a.shared_expert_ff, shared);
    f += check("fa_interval",    a.full_attention_interval, fa_int);
    f += check("is_moe",         a.is_moe ? 1 : 0, is_moe);
    f += check("is_hybrid",      a.is_hybrid ? 1 : 0, is_hybrid);
    /* If DeepSeek V4, print MLA dims */
    if (a.arch == WUBU_ARCH_DEEPSEEK_V4_MOE) {
        printf("  MLA dims: q_lora=%d, kv_lora=%d, rope_head_dim=%d, head_dim_full=%d\n",
               a.q_lora_rank, a.kv_lora_rank, a.rope_head_dim, a.head_dim_full);
    }
    /* hybrid layer_types: every fa_interval-th layer must be full_attention */
    if (is_hybrid && a.full_attention_interval > 0) {
        int bad = 0;
        for (int l = 0; l < a.n_layers; l++) {
            int full = (l % a.full_attention_interval == a.full_attention_interval - 1) ? 1 : 0;
            if (a.layer_types[l] != full) bad++;
        }
        printf("  %-22s layer_types pattern %s\n", "hybrid pattern",
               bad == 0 ? "OK" : "FAIL");
        if (bad) f++;
    }
    return f;
}

int main(void) {
    const char *md = models_dir();
    char path[1024];
    int fails = 0, skipped = 0;

    /* KAT-Coder: skip if not downloaded yet (persistent dir absent). */
    snprintf(path, sizeof(path), "%s/KAT-Coder-V2.5-Dev/config.json", md);
    if (!exists(path)) {
        printf("=== KAT-Coder-V2.5-Dev ===\n  SKIP: config absent at %s (download pending)\n", path);
        skipped++;
    } else {
        fails += test_one(path, "KAT-Coder-V2.5-Dev",
            2048, 40, 256, 8, 16, 32, 128, 4, 512, 4, 1, 1);
    }

    /* Qwen3.6-27B: dir is "Qwen3.6-27B" on persistent storage (not "-base"). */
    snprintf(path, sizeof(path), "%s/Qwen3.6-27B/config.json", md);
    if (!exists(path)) {
        printf("=== Qwen3.6-27B ===\n  SKIP: config absent at %s (download pending)\n", path);
        skipped++;
    } else {
        fails += test_one(path, "Qwen3.6-27B",
            5120, 64, 0, 0, 16, 48, 128, 4, 0, 4, 0, 1);
    }

    /* DeepSeek-V4-Flash: 284B MoE, MLA attention, 129280 vocab */
    snprintf(path, sizeof(path), "%s/DeepSeek-V4-Flash-0731-GGUF/config.json", md);
    if (!exists(path)) {
        printf("=== DeepSeek-V4-Flash ===\n  SKIP: config absent at %s (download pending)\n", path);
        skipped++;
    } else {
        fails += test_one(path, "DeepSeek-V4-Flash",
            7168, 43, 256, 6, 0, 0, 0, 4, 18432, 0, 1, 0);
    }

    /* LFM2.5: 2.6B dense hybrid Mamba2 */
    snprintf(path, sizeof(path), "%s/LFM2.5-2.6B/config.json", md);
    if (!exists(path)) {
        printf("=== LFM2.5-2.6B ===\n  SKIP: config absent at %s (download pending)\n", path);
        skipped++;
    } else {
        fails += test_one(path, "LFM2.5-2.6B",
            2048, 30, 0, 0, 0, 0, 0, 4, 10752, 0, 0, 1);
    }

    if (fails) { printf("\nFAIL: %d checks wrong (%d skipped)\n", fails, skipped); return 1; }
    printf("\nPASS: adapter derives real model dims from config.json (%d skipped, absent on disk)\n", skipped);
    return 0;
}
