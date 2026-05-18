#include "gguf_reader.h"
#include <stdio.h>
int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    // Print types of MoE-related tensors for layer 0
    const char *names[] = {"blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_exps.weight",
                           "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight",
                           "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight",
                           "blk.0.ffn_down_shexp.weight", "output.weight",
                           "token_embd.weight",
                           "blk.3.attn_q.weight", "blk.3.attn_k.weight",
                           "blk.3.attn_v.weight", "blk.3.attn_output.weight",
                           "blk.3.attn_q_norm.weight", "blk.3.attn_k_norm.weight",
                           NULL};
    const char *typestr[] = {"F32","F16","Q4_0","Q4_1","??","??","Q5_0","Q5_1",
        "Q8_0","Q8_1","Q2_K","Q3_K","Q4_K","Q5_K","Q6_K","Q8_K",
        "IQ2_XXS","IQ2_XS","IQ3_XXS","IQ1_S","??","IQ3_S","IQ2_S","IQ1_M"};
    for (int i = 0; names[i]; i++) {
        gguf_tensor_info *t = gguf_find_tensor(ctx, names[i]);
        if (!t) { printf("%s: NOT FOUND\n", names[i]); continue; }
        int64_t ne = 1;
        for (int d = 0; d < t->n_dims; d++) ne *= t->dims[d];
        const char *ts = "UK";
        if (t->ggml_type >= 0 && t->ggml_type < 24) ts = typestr[t->ggml_type];
        printf("%s: type=%d(%s) dims=[%ld", names[i], t->ggml_type, ts, (long)t->dims[0]);
        for (int d = 1; d < t->n_dims; d++) printf(",%ld", (long)t->dims[d]);
        printf("] n_elems=%ld\n", (long)ne);
    }
    gguf_close(ctx);
    return 0;
}
