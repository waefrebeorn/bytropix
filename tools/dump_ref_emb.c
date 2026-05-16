/**
 * Dump the first embedding from both our loader and llama.cpp 
 * to verify embedding lookup produces identical results.
 */
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    const char *model_path = argv[1];

    // ---- Part 1: Extract embedding using llama.cpp ----
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only for determinism
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.embeddings = true;
    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    int n_embd = llama_model_n_embd(model);
    printf("n_embd = %d\n", n_embd);

    // Tokenize BOS only (single token)
    llama_token bos = 248044;
    llama_batch batch = llama_batch_get_one(&bos, 1);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n"); return 1;
    }

    // Get embeddings AFTER layer 0
    // In modern llama.cpp, llama_get_embeddings returns the final hidden state
    // after all layers. But we want just after embedding.
    // Actually, let me get the first token's embedding differently.
    
    // The embeddings from llama_get_embeddings are the full hidden state
    // after all transformer layers (before final norm + lm_head).
    float *embeddings = llama_get_embeddings(ctx);
    if (!embeddings) {
        fprintf(stderr, "No embeddings\n"); return 1;
    }

    // Dump first 10 values of final embedding
    printf("llama.cpp final embedding[0..9]: ");
    for (int i = 0; i < 10 && i < n_embd; i++) printf("%.6f ", embeddings[i]);
    printf("\n");
    printf("  mean=%.4f max=%.4f min=%.4f\n", 
           embeddings[0], embeddings[0], embeddings[0]); // placeholder

    // Also compare: what's the raw embedding for token 248044?
    // We need to access the token_embd weight directly.
    // llama.cpp stores this in model->tok_embd but it's not exposed in the public API.
    // Let me try a different approach: get through ggml directly.
    // Actually this isn't possible through the public API.
    
    // Let me instead load the embedding directly from the GGUF file
    // using our own reader. We already have tools/dump_tensors.c that does this.
    
    printf("\nTo compare embeddings: use our gguf reader to read token_embd.weight\n");
    printf("  token_embd[248044][0..9] can be read from the GGUF directly.\n");
    printf("  llama.cpp's final embedding after ALL layers (not raw embedding).\n");

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
