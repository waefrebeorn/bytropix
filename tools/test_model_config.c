/*
 * test_model_config.c -- verify the HF adapter derives the REAL model dims
 * from config.json for the four target models. Memory-safe: reads only the
 * small config.json, never opens the weights.
 */
#include "wubu_model_adapter.h"
#include <stdio.h>
#include <string.h>

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
    int fails = 0;
    fails += test_one("/tmp/models/KAT-Coder-V2.5-Dev/config.json", "KAT-Coder-V2.5-Dev",
        2048, 40, 256, 8, 16, 32, 128, 4, 512, 4, 1, 1);
    fails += test_one("/tmp/models/Qwen3.6-27B-base/config.json", "Qwen3.6-27B-base",
        5120, 64, 0, 0, 16, 48, 128, 4, 0, 4, 0, 1);
    if (fails) { printf("\nFAIL: %d checks wrong\n", fails); return 1; }
    printf("\nPASS: adapter derives real KAT + Qwen3.6 dims from config.json\n");
    return 0;
}
