/* Dump reference BOS embedding and output.weight from llama.cpp */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama.h"

int main(int argc, char **argv) {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    llama_backend_init();
    
    struct llama_model_params lmp = llama_model_default_params();
    lmp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(path, lmp);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int n_embd = llama_model_n_embd(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    printf("n_embd=%d n_vocab=%d\n", n_embd, n_vocab);
    
    // Create context with embedding output
    struct llama_context_params lcp = llama_context_default_params();
    lcp.n_ctx = 128;
    lcp.embeddings = true; // Enable embedding output
    struct llama_context *ctx = llama_init_from_model(model, lcp);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }
    
    // Decode BOS token
    llama_token bos = llama_vocab_bos(vocab);
    llama_token tokens[1] = {bos};
    struct llama_batch batch = llama_batch_get_one(tokens, 1);
    if (llama_decode(ctx, batch)) { fprintf(stderr, "Decode failed\n"); return 1; }
    
    // Get hidden state (embeddings)
    float *emb = llama_get_embeddings_ith(ctx, 0);
    if (emb) {
        FILE *f = fopen("/tmp/ref_bos_emb.bin", "wb");
        fwrite(emb, sizeof(float), n_embd, f);
        fclose(f);
        float em = 0, es = 0;
        for (int i = 0; i < n_embd; i++) em += emb[i];
        em /= n_embd;
        for (int i = 0; i < n_embd; i++) es += (emb[i]-em)*(emb[i]-em);
        es = sqrtf(es/n_embd);
        printf("Ref BOS hidden: mean=%.6f std=%.6f\n", em, es);
    } else {
        printf("No embeddings available\n");
    }
    
    // Get logits (can verify these)
    float *logits = llama_get_logits_ith(ctx, 0);
    if (logits) {
        FILE *f = fopen("/tmp/ref_logits.bin", "wb");
        fwrite(logits, sizeof(float), n_vocab, f);
        fclose(f);
        printf("Logits saved\n");
    }
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    printf("Done\n");
    return 0;
}
