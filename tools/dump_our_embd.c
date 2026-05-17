#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gguf_reader.h"

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) return 1;
    gguf_buffer_data(ctx);

    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t) { fprintf(stderr, "not found\n"); return 1; }
    printf("dims=[%ld,%ld] type=%d\n", (long)t->dims[0], (long)t->dims[1], t->ggml_type);

    int64_t d_model = t->dims[0];
    int64_t n_vocab = t->dims[1];
    int64_t raw_per_tok = gguf_raw_size(t->ggml_type, d_model);
    printf("raw_per_tok=%ld\n", (long)raw_per_tok);

    float *embd = (float *)malloc(d_model * sizeof(float));

    for (int tok = 0; tok < 5; tok++) {
        gguf_tensor_info fake = *t;
        fake.dims[0] = d_model;
        fake.dims[1] = 1;
        fake.data_offset = t->data_offset + tok * raw_per_tok;
        gguf_read_tensor_f32(ctx, &fake, embd, d_model);
        float s = 0;
        for (int i = 0; i < d_model; i++) s += embd[i] * embd[i];
        printf("Token %d: first 5: %.4f %.4f %.4f %.4f %.4f rms=%.4f\n",
               tok, embd[0], embd[1], embd[2], embd[3], embd[4],
               sqrtf(s / d_model));
    }

    {
        gguf_tensor_info fake = *t;
        fake.dims[0] = d_model;
        fake.dims[1] = 1;
        fake.data_offset = t->data_offset + 9419 * raw_per_tok;
        gguf_read_tensor_f32(ctx, &fake, embd, d_model);
        printf("Token 9419 'Hello': first 10: ");
        for (int i = 0; i < 10; i++) printf("%.4f ", embd[i]);
        float s = 0;
        for (int i = 0; i < d_model; i++) s += embd[i] * embd[i];
        printf("\n  rms=%.4f\n", sqrtf(s / d_model));
        FILE *f = fopen("/tmp/our_token9419_embd.bin", "wb");
        if (f) { fwrite(embd, sizeof(float), d_model, f); fclose(f); }
    }

    free(embd);
    gguf_close(ctx);
    return 0;
}
