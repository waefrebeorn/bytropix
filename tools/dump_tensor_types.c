#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) return 1;
    
    const char *names[] = {
        "token_embd.weight", "output_norm.weight", "output.weight",
        "blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_beta.weight",
        "blk.0.ssm_alpha.weight", "blk.0.ssm_norm.weight", "blk.0.ssm_out.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight",
        "blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight",
        "blk.3.attn_output.weight",
        "blk.3.attn_q_norm.weight", "blk.3.attn_k_norm.weight",
        "blk.0.ffn_gate_exps.weight", "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_gate_inp.weight"
    };
    
    for (int i = 0; i < sizeof(names)/sizeof(names[0]); i++) {
        gguf_tensor_info *t = gguf_find_tensor(ctx, names[i]);
        if (t) {
            const char *typs[] = {"f32","f16","q4_0","q4_1","q5_0","q5_1","q8_0","q8_1",
                "q2_K","q3_K","q4_K","q5_K","q6_K","q8_K","q2_K","q3_K"};
            printf("%-40s type=%-2d", names[i], t->ggml_type);
            if (t->ggml_type == 0) printf("(f32)");
            else if (t->ggml_type == 1) printf("(f16)");
            else if (t->ggml_type == 10) printf("(q4_K)");
            else if (t->ggml_type == 11) printf("(q5_K)");
            else if (t->ggml_type == 12) printf("(q6_K)");
            else if (t->ggml_type == 13) printf("(q4_K)");
            else if (t->ggml_type == 16) printf("(q3_K)");
            else if (t->ggml_type == 22) printf("(iq2_xxs)");
            else if (t->ggml_type == 27) printf("(iq2_M)");
            else if (t->ggml_type == 28) printf("(iq4_xs)");
            else if (t->ggml_type == 20) printf("(iq3_xxs)");
            else printf("(type%d)", t->ggml_type);
            printf(" dims=[");
            for (int j = 0; j < t->n_dims; j++) {
                if (j > 0) printf(",");
                printf("%ld", t->dims[j]);
            }
            printf("]\n");
        }
    }
    gguf_close(ctx);
    return 0;
}
