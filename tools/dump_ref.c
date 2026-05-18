// dump_ref.c — Dump reference logits and per-layer hidden states from llama.cpp
// Compile: gcc -o /tmp/dump_ref dump_ref.c -I/home/wubu/llama.cpp -I/home/wubu/llama.cpp/common
//   -I/home/wubu/llama.cpp/ggml/include -L/home/wubu/llama.cpp/build/bin
//   -Wl,-rpath,/home/wubu/llama.cpp/build/bin -lggml-cpu
//   -L/home/wubu/llama.cpp/build -l:libllama.a -lm -lpthread -ldl -lstdc++
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "llama.h"

#define D_MODEL 2048

int main(int argc, char **argv) {
    const char *model_path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    // Initialize llama backend
    llama_backend_init();
    
    // Model params
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    
    // Load model
    struct llama_model *model = llama_load_model_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    // Create context
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 1;
    
    struct llama_context *ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_free_model(model);
        return 1;
    }
    
    // Prepare input: token 248044
    llama_token token = 248044;
    llama_batch batch = llama_batch_get_one(&token, 1);
    
    // Run forward pass
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        llama_free(ctx);
        llama_free_model(model);
        return 1;
    }
    
    // Get logits
    float *logits = llama_get_logits(ctx);
    int n_vocab = llama_n_vocab(model);
    printf("llama.cpp output: n_vocab=%d\n", n_vocab);
    
    // Dump logits
    FILE *f = fopen("/tmp/llama_logits_new.bin", "wb");
    if (f) {
        fwrite(logits, sizeof(float), n_vocab, f);
        fclose(f);
        printf("  Logits saved to /tmp/llama_logits_new.bin (%.1f MB)\n",
               n_vocab * 4.0 / 1024 / 1024);
    }
    
    // Print top-10
    float *cpy = (float *)malloc(n_vocab * sizeof(float));
    memcpy(cpy, logits, n_vocab * sizeof(float));
    printf("  Top-10 logits:\n");
    for (int k = 0; k < 10; k++) {
        float best = -1e30f; int best_idx = -1;
        for (int i = 0; i < n_vocab; i++) {
            if (cpy[i] > best) { best = cpy[i]; best_idx = i; }
        }
        cpy[best_idx] = -1e30f;
        printf("    [%d] val=%.4f\n", best_idx, (double)best);
    }
    free(cpy);
    
    // Dump hidden states from each layer
    // (llama.cpp doesn't easily expose per-layer hidden states via public API,
    //  so we skip this for now)
    
    printf("\nNOTE: Per-layer hidden states from llama.cpp require source modification.\n");
    printf("Skipping per-layer comparison for now.\n");
    
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
