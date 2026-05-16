/**
 * Compare weight loading between our gguf_reader and llama.cpp.
 * Load ssm_beta.weight (F32) from the GGUF using both and compare.
 */
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Our gguf_reader
#include "gguf_reader.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }
    const char *model_path = argv[1];

    // ---- Part 1: Load with llama.cpp and get tensor directly ----
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed llama load\n"); return 1; }

    // We can't easily access raw tensors from llama.cpp public API.
    // But we can create a context and decode a single token, then check.
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 64;
    cparams.n_batch = 64;
    cparams.embeddings = true;
    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed ctx\n"); return 1; }

    int n_embd = llama_model_n_embd(model);

    // ---- Part 2: Load with our gguf_reader ----
    gguf_ctx *gctx = gguf_open(model_path);
    if (!gctx) { fprintf(stderr, "Failed gguf_open\n"); return 1; }

    // Read ssm_beta.weight from our reader (dims=[2048, 32])
    gguf_tensor_info *t = gguf_find_tensor(gctx, "blk.0.ssm_beta.weight");
    if (!t) { fprintf(stderr, "ssm_beta.weight not found\n"); return 1; }
    
    float *our_beta = (float *)malloc(2048 * 32 * sizeof(float));
    int nread = gguf_read_tensor_f32(gctx, t, our_beta, 2048 * 32);
    if (nread <= 0) { fprintf(stderr, "Failed to read our beta\n"); return 1; }
    printf("Our ssm_beta: %d elements read\n", nread);

    // ---- Part 3: Now run a single token through both and compare the QKV projection ----
    // This is the most important test.
    
    // Get BOS embedding from both implementations
    // For ours: read from token_embd.weight directly
    gguf_tensor_info *te = gguf_find_tensor(gctx, "token_embd.weight");
    if (!te) { fprintf(stderr, "token_embd not found\n"); return 1; }
    
    int bos = 248044;
    float *our_emb = (float *)malloc(n_embd * sizeof(float));
    
    // Read the BOS embedding row. dims=[2048, 248320], row 'bos' starts at bos * n_embd
    // We need to read through gguf_read_tensor_f32 which dequantizes
    // But reading the whole embedding (248320*2048 floats) is too big.
    // Instead, let's read the embedding weight from file offset directly.
    // Actually, we can read just one row by seeking and reading raw data.
    
    // For now, let's just compare the beta weight values (F32, no dequant)
    printf("\nComparing ssm_beta.weight (first 10 values):\n");
    printf("Index  Our_Beta\n");
    for (int i = 0; i < 10; i++) {
        printf("%5d  %.10f\n", i, our_beta[i]);
    }
    
    // Also dump beta for comparison by running llama.cpp
    // Use a single token forward to get embeddings
    llama_token token = bos;
    llama_batch batch = llama_batch_get_one(&token, 1);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n"); return 1;
    }
    
    float *embeddings = llama_get_embeddings(ctx);
    if (!embeddings) { fprintf(stderr, "No embeddings\n"); return 1; }
    
    // Dump llama.cpp's first embedding values
    printf("\nllama.cpp final embedding for BOS (after ALL layers):\n");
    printf("Index  Value\n");
    for (int i = 0; i < 10; i++) {
        printf("%5d  %.10f\n", i, embeddings[i]);
    }
    
    // Compare with our beta_raw for the same token
    // We need to run just the beta projection. This requires loading the full model.
    // For now, let's just check the token_embd weight entry for token 0
    printf("\nChecking token 0 embedding from GGUF (our reader):\n");
    // dims=[2048, 248320], row 0 starts at offset 0
    // With ne[0]=2048, ne[1]=248320, the data is stored row-major with ne[0] varying fastest
    // Row t: t * ne[0] = t * 2048
    // We can read just the first row by reading 2048 floats from the start
    // But the tensor is quantized (Q5_K = type 13), so we need dequantization
    
    // Let's read all of token_embd (slow but works for verification)
    // Actually, let's just check by computing the embedding for token 0 manually
    
    gguf_close(gctx);
    llama_free(ctx);
    llama_model_free(model);
    free(our_beta);
    free(our_emb);
    return 0;
}
