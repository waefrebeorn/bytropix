#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gguf_reader.h"

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) return 1;
    gguf_buffer_data(ctx);

    // Load token embeddings for tokens 0-10 + some known tokens
    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t) { fprintf(stderr, "token_embd.weight not found\n"); return 1; }
    printf("dims=[%ld,%ld] type=%d\n", (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    
    int64_t n_embd = t->dims[0];  // 2048
    int64_t n_vocab = t->dims[1]; // 248320
    
    float *embd = (float *)malloc(n_embd * sizeof(float));
    // Read token 9419 ("Hello")
    gguf_tensor_info t_one = *t;
    t_one.dims[1] = 1;  // just 1 token
    t_one.data_offset = t->data_offset + 9419 * gguf_raw_size(t->ggml_type, n_embd);
    int ret = gguf_read_tensor_f32(ctx, &t_one, embd, n_embd);
    printf("Token 9419 '%s': read %d floats\n", "Hello", ret);
    printf("  first 10: ");
    for (int i = 0; i < 10; i++) printf("%.4f ", embd[i]);
    printf("\n  rms=%.4f\n", sqrtf(n_embd > 0 ? ({
        float s=0; for(int i=0;i<n_embd;i++) s+=embd[i]*embd[i]; s/n_embd;
    }) : 0));

    // Also check via the full read
    float *full = (float *)malloc(n_embd * n_vocab / 1000 * sizeof(float)); // partial
    // Just read first few tokens
    float *first_tokens = (float *)malloc(10 * n_embd * sizeof(float));
    gguf_tensor_info t_part = *t;
    t_part.dims[1] = 10;
    t_part.data_offset = t->data_offset;
    ret = gguf_read_tensor_f32(ctx, &t_part, first_tokens, 10 * n_embd);
    printf("\nFirst 10 tokens (first 5 dims each):\n");
    for (int tok = 0; tok < 10; tok++) {
        printf("  [%d]: ", tok);
        for (int d = 0; d < 5; d++)
            printf("%.4f ", first_tokens[tok * n_embd + d]);
        printf("\n");
    }

    free(embd); free(full); free(first_tokens);
    gguf_close(ctx);
    return 0;
}
