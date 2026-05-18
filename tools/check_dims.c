/* Verify gguf_reader tensor dimensions */
#include <stdio.h>
#include "gguf_reader.h"

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    
    // Dump key tensors
    const char *names[] = {
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_dt.bias",
        "blk.0.ssm_a",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_norm.weight",
        "blk.0.ssm_out.weight",
        "blk.0.attn_norm.weight",
        "blk.0.post_attention_norm.weight",
    };
    
    for (int i = 0; i < sizeof(names)/sizeof(names[0]); i++) {
        gguf_tensor_info *t = gguf_find_tensor(ctx, names[i]);
        if (t) {
            printf("%s: dims=[", names[i]);
            int64_t total = 1;
            for (int d = 0; d < t->n_dims; d++) {
                printf("%ld%s", (long)t->dims[d], d+1<t->n_dims ? ", " : "");
                total *= t->dims[d];
            }
            printf("] total=%ld type=%d offset=%lu\n", (long)total, t->ggml_type, (unsigned long)t->data_offset);
        } else {
            printf("%s: NOT FOUND\n", names[i]);
        }
    }
    
    gguf_close(ctx);
    return 0;
}
