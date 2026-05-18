#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { printf("FAIL: gguf_open\n"); return 1; }
    
    int total = ctx->n_tensors;
    
    const char *type_names[] = {
        "F32","F16","BF16","Q4_0","Q4_1","Q5_0","Q5_1","Q8_0",
        "Q8_1","Q2_K","Q3_K","Q4_K","Q5_K","Q6_K","Q8_K","IQ2_XXS",
        "IQ2_XS","IQ3_XXS","IQ3_XS","IQ4_XS","IQ1_S","IQ4_NL","Q3_S","Q3_S_XL",
        "IQ3_S","IQ3_M","IQ2_M","IQ2_S","IQ1_M","BF16","Q4_0","Q4_1"
    };
    
    int counts[32] = {0};
    for (int i = 0; i < total; i++) {
        int t = ctx->tensors[i].ggml_type;
        if (t >= 0 && t < 32) counts[t]++;
    }
    
    printf("Total tensors: %d\n\nType counts:\n", total);
    for (int i = 0; i < 32; i++) {
        if (counts[i] > 0) printf("  %-12s: %d\n", type_names[i], counts[i]);
    }
    
    printf("\n--- SSM attn_qkv type (L0-L2) ---\n");
    for (int layer = 0; layer < 3; layer++) {
        char name[256]; snprintf(name,256,"blk.%d.attn_qkv.weight",layer);
        gguf_tensor_info *t = gguf_find_tensor(ctx, name);
        if (t) printf("  blk.%d attn_qkv: type=%d dims=[%lld,%lld]\n",
               layer, t->ggml_type, (long long)t->dims[0], (long long)t->dims[1]);
    }
    
    printf("\n--- GQA attn_qkv type (L3,7,11,15,19,23,27,31,35,39) ---\n");
    int gqa[] = {3,7,11,15,19,23,27,31,35,39};
    for (int li = 0; li < 10; li++) {
        char name[256]; snprintf(name,256,"blk.%d.attn_qkv.weight",gqa[li]);
        gguf_tensor_info *t = gguf_find_tensor(ctx, name);
        if (t) printf("  blk.%d attn_qkv: type=%d\n", gqa[li], t->ggml_type);
    }
    
    printf("\n--- MoE expert types (L0) ---\n");
    const char *moe_tensors[] = {
        "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_up_shexp.weight",
        "blk.0.ffn_down_shexp.weight",
        "blk.0.ffn_gate_inp_shexp.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight"
    };
    for (int i = 0; i < 8; i++) {
        gguf_tensor_info *t = gguf_find_tensor(ctx, moe_tensors[i]);
        if (t) {
            printf("  %s: type=%d n_dims=%d dims=[%lld,%lld", 
                   moe_tensors[i], t->ggml_type, t->n_dims,
                   (long long)t->dims[0], (long long)t->dims[1]);
            if (t->n_dims > 2) printf(",%lld", (long long)t->dims[2]);
            printf("]\n");
        } else {
            printf("  %s: NOT FOUND\n", moe_tensors[i]);
        }
    }
    
    // Check for fused gate_up 
    printf("\nChecking ffn_gate_up_exps at L0, L17, L39:\n");
    int check_layers[] = {0, 17, 39};
    for (int li = 0; li < 3; li++) {
        char name[256]; snprintf(name,256,"blk.%d.ffn_gate_up_exps.weight",check_layers[li]);
        gguf_tensor_info *t = gguf_find_tensor(ctx, name);
        if (t) printf("  blk.%d gate_up_exps (FUSED): type=%d dims=[%lld,%lld,%lld]\n",
               check_layers[li], t->ggml_type,
               (long long)t->dims[0], (long long)t->dims[1], (long long)t->dims[2]);
        else printf("  blk.%d: NOT FOUND (using separate gate/up)\n", check_layers[li]);
    }
    
    printf("\n--- down_exps type across layers ---\n");
    for (int layer = 0; layer < 40; layer++) {
        char name[256]; snprintf(name,256,"blk.%d.ffn_down_exps.weight",layer);
        gguf_tensor_info *t = gguf_find_tensor(ctx, name);
        if (t) printf("  blk.%d: type=%d\n", layer, t->ggml_type);
    }
    
    printf("\n--- output.weight ---\n");
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    if (t) printf("  output.weight: type=%d dims=[%lld,%lld]\n", t->ggml_type,
           (long long)t->dims[0], (long long)t->dims[1]);
    
    printf("\n--- token_embd.weight ---\n");
    t = gguf_find_tensor(ctx, "token_embd.weight");
    if (t) printf("  token_embd.weight: type=%d dims=[%lld,%lld]\n", t->ggml_type,
           (long long)t->dims[0], (long long)t->dims[1]);
    
    printf("\n--- ssm_out weight (L0, L2, L3(GQA), L4) ---\n");
    for (int layer = 0; layer < 5; layer++) {
        char name[256]; snprintf(name,256,"blk.%d.ssm_out.weight",layer);
        t = gguf_find_tensor(ctx, name);
        if (t) printf("  blk.%d: type=%d dims=[%lld,%lld]\n", layer, t->ggml_type,
               (long long)t->dims[0], (long long)t->dims[1]);
    }
    
    printf("\n--- attn_gate weight type (blk.0, 3, 7, 11) ---\n");
    int check[] = {0,3,7,11};
    for (int li = 0; li < 4; li++) {
        char name[256]; snprintf(name,256,"blk.%d.attn_gate.weight",check[li]);
        t = gguf_find_tensor(ctx, name);
        if (t) printf("  blk.%d attn_gate: type=%d dims=[%lld,%lld]\n",
               check[li], t->ggml_type, (long long)t->dims[0], (long long)t->dims[1]);
        else printf("  blk.%d attn_gate: NOT FOUND\n", check[li]);
    }
    
    printf("\n--- GQA attn_q/k/v weights (L3) ---\n");
    const char *qkv[] = {"attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight",
                         "attn_q_norm.weight","attn_k_norm.weight"};
    for (int i = 0; i < 6; i++) {
        char name[256]; snprintf(name,256,"blk.3.%s",qkv[i]);
        t = gguf_find_tensor(ctx, name);
        if (t) printf("  blk.3.%s: type=%d dims=[%lld,%lld]\n", 
               qkv[i], t->ggml_type, (long long)t->dims[0], (long long)t->dims[1]);
        else printf("  blk.3.%s: NOT FOUND\n", qkv[i]);
    }
    
    gguf_close(ctx);
    return 0;
}
