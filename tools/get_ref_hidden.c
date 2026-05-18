/* Get reference last hidden state from llama.cpp */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "llama.h"

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    llama_backend_init();
    
    struct llama_model_params lmp = llama_model_default_params();
    lmp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(path, lmp);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    
    struct llama_context_params lcp = llama_context_default_params();
    lcp.n_ctx = 128;
    lcp.embeddings = true;  // Enable embedding mode to get hidden states
    struct llama_context *ctx = llama_init_from_model(model, lcp);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }
    
    llama_token bos = llama_vocab_bos(llama_model_get_vocab(model));
    llama_token tokens[1] = {bos};
    struct llama_batch batch = llama_batch_get_one(tokens, 1);
    if (llama_decode(ctx, batch)) { fprintf(stderr, "Decode failed\n"); return 1; }
    
    // Get last hidden state (only available with embeddings=true)
    float *embd = llama_get_embeddings_ith(ctx, 0);
    if (embd) {
        int n_embd = llama_model_n_embd(model);
        printf("Got hidden state: n_embd=%d\n", n_embd);
        
        double m=0, s=0;
        for (int i = 0; i < 10; i++) printf("  h[%d]=%.6f\n", i, embd[i]);
        for (int i = 0; i < n_embd; i++) { m += embd[i]; s += embd[i]*embd[i]; }
        printf("mean=%.6f std=%.6f\n", m/n_embd, sqrt(s/n_embd - (m/n_embd)*(m/n_embd)));
        
        FILE *f = fopen("/tmp/ref_hidden.bin", "wb");
        fwrite(embd, sizeof(float), n_embd, f);
        fclose(f);
        printf("Saved hidden state to /tmp/ref_hidden.bin\n");
    } else {
        printf("No embeddings available - llama_get_embeddings_ith returned NULL\n");
    }
    
    // Also get logits
    float *logits = llama_get_logits_ith(ctx, 0);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    double lm=0;
    for (int i = 0; i < 50000; i++) lm += logits[i];
    printf("Logits (first 50000): mean=%.6f\n", lm/50000);
    
    FILE *f = fopen("/tmp/ref_logits_final.bin", "wb");
    fwrite(logits, sizeof(float), n_vocab, f);
    fclose(f);
    printf("Logits saved\n");
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
