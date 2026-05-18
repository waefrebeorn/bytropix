#include "gguf_reader.h"
#include <stdio.h>
#include <math.h>

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) return 1;

    // Check ssm_a values (F32, 32 elements)
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.ssm_a");
    if (!t) { fprintf(stderr, "ssm_a not found\n"); return 1; }
    printf("ssm_a: dims=[%ld] type=%d\n", (long)t->dims[0], t->ggml_type);

    float vals[32];
    int n = gguf_read_tensor_f32(ctx, t, vals, 32);
    printf("Read %d elements\n", n);
    printf("ssm_a[0:8]:");
    for (int i = 0; i < 8 && i < 32; i++) printf(" %+.6f", vals[i]);
    printf("\n");
    printf("ssm_a min=%.6f max=%.6f\n", vals[0], vals[31]);

    // Check ssm_dt.bias (F32, 32 elements)
    t = gguf_find_tensor(ctx, "blk.0.ssm_dt.bias");
    if (t) {
        gguf_read_tensor_f32(ctx, t, vals, 32);
        printf("\nssm_dt.bias[0:8]:");
        for (int i = 0; i < 8; i++) printf(" %+.6f", vals[i]);
        printf("\n");
    }

    // Check ssm_alpha.weight
    t = gguf_find_tensor(ctx, "blk.0.ssm_alpha.weight");
    if (t) {
        float *alpha = (float*)malloc(2048 * 32 * sizeof(float));
        gguf_read_tensor_f32(ctx, t, alpha, 2048 * 32);
        printf("\nssm_alpha.weight[0,0:4]:");
        for (int i = 0; i < 4; i++) printf(" %+.6f", alpha[i]);
        printf("\n");
        double mean = 0;
        for (int i = 0; i < 2048 * 32; i++) mean += alpha[i];
        mean /= (2048 * 32);
        printf("ssm_alpha.weight mean=%f\n", mean);
        free(alpha);
    }

    // Check embedding sample
    t = gguf_find_tensor(ctx, "token_embd.weight");
    if (t) {
        float *emb = (float*)malloc(2048 * sizeof(float));
        gguf_read_tensor_f32(ctx, t, emb, 2048);
        printf("\ntoken_embd.weight[0,0:4]:");
        for (int i = 0; i < 4; i++) printf(" %+.6f", emb[i]);
        printf("\n");
        free(emb);
    }

    gguf_close(ctx);
    return 0;
}
