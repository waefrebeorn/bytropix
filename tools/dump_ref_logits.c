/* Dump reference logits from llama.cpp for a single BOS token */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama.h"

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    llama_backend_init();
    
    struct llama_model_params lmp = llama_model_default_params();
    lmp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(path, lmp);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    int n_embd = llama_model_n_embd(model);
    printf("llama: n_vocab=%d n_embd=%d n_layer=%d\n", 
           n_vocab, n_embd, llama_model_n_layer(model));
    
    struct llama_context_params lcp = llama_context_default_params();
    lcp.n_ctx = 128;
    struct llama_context *ctx = llama_init_from_model(model, lcp);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }
    
    llama_token bos = llama_vocab_bos(vocab);
    llama_token tokens[1] = {bos};
    struct llama_batch batch = llama_batch_get_one(tokens, 1);
    if (llama_decode(ctx, batch)) { fprintf(stderr, "Decode failed\n"); return 1; }
    
    float *logits = llama_get_logits_ith(ctx, 0);
    
    FILE *f = fopen("/tmp/ref_logits.bin", "wb");
    fwrite(logits, sizeof(float), n_vocab, f);
    fclose(f);
    
    int top[5] = {0}; float tv[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
    for (int i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (v > tv[4]) {
            tv[4] = v; top[4] = i;
            for (int k = 3; k >= 0; k--) {
                if (tv[k] < tv[k+1]) {
                    float t = tv[k]; tv[k] = tv[k+1]; tv[k+1] = t;
                    int ti = top[k]; top[k] = top[k+1]; top[k+1] = ti;
                }
            }
        }
    }
    printf("llama top-5:\n");
    for (int k = 0; k < 5; k++)
        printf("  [%d] val=%.4f\n", top[k], tv[k]);
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    printf("Done. ref_logits saved to /tmp/ref_logits.bin\n");
    return 0;
}
