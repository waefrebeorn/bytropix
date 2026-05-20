/**
 * Quick check: what GGML types are used for SSM weights?
 */
#include "gguf_reader.h"
#include <stdio.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;

    printf("bytropix GGML types: Q5_K=%d Q6_K=%d IQ2_XXS=%d IQ3_XXS=%d IQ4_XS=%d Q8_0=%d\n",
        GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0);

    const char *names[] = {
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_out.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "token_embd.weight",
        "output.weight"
    };

    for (int i = 0; i < sizeof(names)/sizeof(names[0]); i++) {
        gguf_tensor_info *t = gguf_find_tensor(ctx, names[i]);
        if (t) {
            printf("%-35s type=%d dims=[", names[i], t->ggml_type);
            for (int d = 0; d < t->n_dims; d++)
                printf("%s%ld", d ? "," : "", (long)t->dims[d]);
            printf("]\n");
        }
    }
    gguf_close(ctx);
    return 0;
}
