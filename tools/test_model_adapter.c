/* test_model_adapter.c -- verify HF config.json parsing for the 4 Colonel models. */
#include "wubu_model_adapter.h"
#include <stdio.h>
#include <string.h>

static int write_cfg(const char *path, const char *json) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(json, 1, strlen(json), f);
    fclose(f);
    return 0;
}

int main(void) {
    // (1) Qwen3.6-27B-ish dense HF config
    const char *c1 =
        "{\"architectures\":[\"Qwen3ForCausalLM\"],\"hidden_size\":5120,"
        "\"num_hidden_layers\":64,\"intermediate_size\":17408,\"num_attention_heads\":24,"
        "\"num_key_value_heads\":4,\"head_dim\":256,\"rope_theta\":10000000,"
        "\"partial_rotary_factor\":0.25}";
    if (write_cfg("cfg_qwen36.json", c1)) return 1;
    wubu_adapter_t a;
    if (!wubu_adapter_load(&a, "cfg_qwen36.json") || !a.ok) { fprintf(stderr, "FAIL: qwen36 load\n"); return 1; }
    if (a.d_model != 5120 || a.n_layers != 64 || a.gqa_kv_heads != 4) { fprintf(stderr, "FAIL: qwen36 dims\n"); return 1; }

    // (2) KAT MoE
    const char *c2 =
        "{\"architectures\":[\"Qwen3ForCausalLM\"],\"hidden_size\":2048,"
        "\"num_hidden_layers\":64,\"num_experts\":35,\"num_experts_per_tok\":3,"
        "\"num_attention_heads\":16}";
    if (write_cfg("cfg_kat.json", c2)) return 1;
    wubu_adapter_t k;
    if (!wubu_adapter_load(&k, "cfg_kat.json") || !k.ok) { fprintf(stderr, "FAIL: kat load\n"); return 1; }
    if (!k.is_moe || k.n_experts != 35 || k.n_active_experts != 3) { fprintf(stderr, "FAIL: kat moe\n"); return 1; }

    // (3) BTL-3 LoRA (has base_model)
    const char *c3 =
        "{\"architectures\":[\"Qwen3ForCausalLM\"],\"base_model\":\"Qwen/Qwen3.6-27B\","
        "\"hidden_size\":5120,\"num_hidden_layers\":64}";
    if (write_cfg("cfg_btl.json", c3)) return 1;
    wubu_adapter_t b;
    if (!wubu_adapter_load(&b, "cfg_btl.json") || !b.ok) { fprintf(stderr, "FAIL: btl load\n"); return 1; }
    if (!b.is_lora || strcmp(b.base_model, "Qwen/Qwen3.6-27B") != 0) { fprintf(stderr, "FAIL: btl lora\n"); return 1; }

    // (4) name resolution w/o file
    wubu_adapter_t n;
    if (!wubu_adapter_resolve_name(&n, "badtheorylabs/BTL-3")) { fprintf(stderr, "FAIL: resolve btl\n"); return 1; }
    if (!n.is_lora) { fprintf(stderr, "FAIL: resolve btl lora flag\n"); return 1; }
    if (!wubu_adapter_resolve_name(&n, "KAT-Coder-V2.5-Dev")) { fprintf(stderr, "FAIL: resolve kat\n"); return 1; }
    if (!n.is_moe) { fprintf(stderr, "FAIL: resolve kat moe\n"); return 1; }

    printf("PASS: adapter (qwen36 / kat-moe / btl-lora / name-resolve)\n");
    return 0;
}
