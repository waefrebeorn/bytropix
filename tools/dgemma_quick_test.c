/** dgemma_quick_test.c — Quick DiffusionGemma forward test */
#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *path = "/home/wubu/models/DiffusionGemma-26B-Q4_K_M.gguf";
    int test_ctx = 64;
    if (argc > 1) test_ctx = atoi(argv[1]);

    printf("=== DiffusionGemma Quick Forward Test ===\n");
    printf("Model: %s\nContext: %d\n\n", path, test_ctx);

    printf("Loading model...\n");
    double t0 = now_sec();
    wubu_model_t model;
    memset(&model, 0, sizeof(model));

    if (!wubu_model_init(&model, path)) {
        fprintf(stderr, "Model init failed\n");
        return 1;
    }
    double t_load = now_sec() - t0;
    printf("Model loaded in %.1fs: %d layers, d_model=%d, vocab=%d\n",
           t_load, model.n_layers, model.d_model, model.vocab_size);

    /* Print per-layer info */
    for (int l = 0; l < model.n_layers; l++) {
        wubu_layer_t *layer = &model.layers[l];
        if (!layer->is_ssm) {
            printf("  Layer %2d: GQA hd=%d qh=%d kh=%d od=%d%s\n",
                   l, layer->gqa.head_dim, layer->gqa.q_heads, layer->gqa.kv_heads,
                   layer->gqa.out_dim, layer->gqa.is_large ? " LARGE" : "");
        }
    }

    /* Run forward pass with random token IDs */
    printf("\nRunning forward pass (B=1, T=%d)...\n", test_ctx);
    int *tokens = (int *)malloc(test_ctx * sizeof(int));
    for (int i = 0; i < test_ctx; i++) tokens[i] = 1000 + (i % 5000);

    float *logits = (float *)calloc((size_t)test_ctx * model.vocab_size, sizeof(float));

    double t_fwd = now_sec();
    wubu_model_forward(&model, tokens, 1, test_ctx, logits);
    double t_fwd_elapsed = now_sec() - t_fwd;

    /* Check output */
    int nan_count = 0;
    float max_logit = -1e30f, min_logit = 1e30f;
    for (int i = 0; i < test_ctx * model.vocab_size; i++) {
        if (isnan(logits[i]) || isinf(logits[i])) nan_count++;
        if (logits[i] > max_logit) max_logit = logits[i];
        if (logits[i] < min_logit) min_logit = logits[i];
    }

    printf("Forward pass: %.3f s (%.1f tok/s)\n", t_fwd_elapsed, test_ctx / t_fwd_elapsed);
    printf("Logits: min=%.4f max=%.4f NaN/Inf=%d / %ld\n",
           min_logit, max_logit, nan_count, (long)(test_ctx * model.vocab_size));

    if (nan_count == 0) {
        printf("\n✓ Forward pass completed successfully — no NaN/Inf\n");
    } else {
        printf("\n✗ Forward pass has %d NaN/Inf values\n", nan_count);
    }

    free(tokens);
    free(logits);
    printf("Calling wubu_model_free...\n");
    wubu_model_free(&model);
    printf("wubu_model_free done\n");

    return (nan_count > 0) ? 1 : 0;
}
