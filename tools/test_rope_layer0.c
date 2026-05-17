/**
 * test_rope_layer0.c — Dump Q before/after RoPE at layer 3 (first GQA layer)
 * Compile: gcc -O2 -I include test_rope_layer0.c src/gguf_reader.o -lm -o test_rope_layer0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_ssm.h"
#include "gguf_reader.h"

#define MAX_CACHE_T 1024
#define GQA_KV_DIM (GQA_KV_HEADS * GQA_HEAD_DIM)

static float *rope_sc = NULL;
static int rope_init(void) {
    if (rope_sc) return 1;
    rope_sc = (float *)malloc((size_t)MAX_CACHE_T * ROTARY_DIM * sizeof(float));
    if (!rope_sc) return 0;
    float theta_scale = powf(ROPE_THETA, -2.0f / ROTARY_DIM);
    for (int p = 0; p < MAX_CACHE_T; p++) {
        float theta = (float)p;
        for (int i = 0; i < ROTARY_DIM / 2; i++) {
            rope_sc[p * ROTARY_DIM + i * 2]     = cosf(theta);
            rope_sc[p * ROTARY_DIM + i * 2 + 1] = sinf(theta);
            theta *= theta_scale;
        }
    }
    return 1;
}

// ADJACENT-pair RoPE (fixed version, correct for Qwen3 MRoPE)
static void apply_rotary_to_buf(float *buf, int n_heads, int position, const float *sc) {
    const float *sc_p = sc + (size_t)position * ROTARY_DIM;
    for (int h = 0; h < n_heads; h++) {
        float *head = buf + (size_t)h * GQA_HEAD_DIM;
        for (int i = 0; i < ROTARY_DIM / 2; i++) {
            float cosv = sc_p[i * 2];
            float sinv = sc_p[i * 2 + 1];
            float d0 = head[i * 2];
            float d1 = head[i * 2 + 1];
            head[i * 2]     = d0 * cosv - d1 * sinv;
            head[i * 2 + 1] = d0 * sinv + d1 * cosv;
        }
    }
}

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    rope_init();

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open GGUF\n"); return 1; }
    gguf_buffer_data(ctx);

    // Load layer 3 (first GQA) K projection weight
    char tname[256];
    snprintf(tname, sizeof(tname), "blk.3.attn_k.weight");
    gguf_tensor_info *ti = gguf_find_tensor(ctx, tname);
    if (!ti) { fprintf(stderr, "Tensor %s not found\n", tname); return 1; }
    
    int64_t n_k = ti->dims[0] * ti->dims[1];
    float *k_weight = (float *)malloc(n_k * sizeof(float));
    gguf_read_tensor_f32(ctx, ti, k_weight, n_k);
    printf("K weight: %ld elems, dims=[%ld,%ld]\n", (long)n_k, (long)ti->dims[0], (long)ti->dims[1]);

    // Also load K norm weight
    snprintf(tname, sizeof(tname), "blk.3.attn_k_norm.weight");
    ti = gguf_find_tensor(ctx, tname);
    if (!ti) { fprintf(stderr, "Tensor %s not found\n", tname); return 1; }
    float *k_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
    gguf_read_tensor_f32(ctx, ti, k_norm_weight, GQA_HEAD_DIM);
    printf("K norm weight: %ld elems\n", (long)GQA_HEAD_DIM);

    // Create a synthetic test input for position 1
    float x[D_MODEL];
    srand(12345);
    for (int i = 0; i < D_MODEL; i++)
        x[i] = (float)(rand() % 1000) / 100.0f - 5.0f;

    // Compute K projection for a single GQA head
    int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
    float k_raw[kv_dim];
    memset(k_raw, 0, sizeof(k_raw));
    for (int j = 0; j < kv_dim; j++) {
        double sum = 0.0;
        for (int i = 0; i < D_MODEL; i++)
            sum += (double)x[i] * (double)k_weight[i + j * D_MODEL];
        k_raw[j] = (float)sum;
    }

    // RMSNorm K
    float k_norm[kv_dim];
    memcpy(k_norm, k_raw, sizeof(k_raw));
    wubu_rms_norm(1, GQA_KV_HEADS, GQA_HEAD_DIM, k_norm, k_norm_weight, 1e-6f, k_norm);

    // Dump K_norm before RoPE at position 1
    printf("\nK_norm BEFORE RoPE (position 1):\n");
    for (int i = 0; i < 16; i++)
        printf(" %.6f", k_norm[i]);
    printf("\n");

    // Save pre-RoPE for reference
    float k_norm_pre[GQA_KV_DIM];
    memcpy(k_norm_pre, k_norm, sizeof(k_norm));

    // Apply RoPE at position 1
    apply_rotary_to_buf(k_norm, GQA_KV_HEADS, 1, rope_sc);

    // Dump K_norm AFTER RoPE at position 1
    printf("\nK_norm AFTER RoPE (adjacent-pair, position 1):\n");
    for (int i = 0; i < 16; i++)
        printf(" %.6f", k_norm[i]);
    printf("\n");

    // Dump full K tensor before and after RoPE for Python verification
    FILE *f1 = fopen("/tmp/k_pre_rope.bin", "wb");
    if (f1) { fwrite(k_norm_pre, sizeof(float), GQA_KV_DIM, f1); fclose(f1); }
    FILE *f2 = fopen("/tmp/k_post_rope.bin", "wb");
    if (f2) { fwrite(k_norm, sizeof(float), GQA_KV_DIM, f2); fclose(f2); }

    printf("\nDumped to /tmp/k_pre_rope.bin and /tmp/k_post_rope.bin\n");

    // Dump the sin/cos table at position 1
    printf("\nSin/cos table at position 1 (first 16 values):\n");
    for (int i = 0; i < 16; i++)
        printf(" %.6f", rope_sc[1 * ROTARY_DIM + i]);
    printf("\n");

    free(k_weight);
    free(k_norm_weight);
    gguf_close(ctx);
    return 0;
}
