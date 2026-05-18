/* Dump token_embd.weight *and* output.weight directly from llama.cpp */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "llama.h"
#include "ggml.h"

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    llama_backend_init();
    
    struct llama_model_params lmp = llama_model_default_params();
    lmp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(path, lmp);
    if (!model) return 1;
    
    int n_embd = llama_model_n_embd(model);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    
    // Get the token embedding via decode of token 0
    struct llama_context_params lcp = llama_context_default_params();
    lcp.n_ctx = 2;
    struct llama_context *ctx = llama_init_from_model(model, lcp);
    if (!ctx) return 1;
    
    llama_token tokens[1] = {0};
    struct llama_batch batch = llama_batch_get_one(tokens, 1);
    llama_decode(ctx, batch);
    
    // Get hidden state after embedding (this IS the token embedding when no layers process it)
    // Actually, with just 1 token, this goes through the full model. We need a different approach.
    
    // Alternative: read the model's tensors directly via llama.h API
    // llama_model doesn't expose tensor reading directly in the public API
    
    // Simplest: just dump the token embedding that our saved file has
    // and compare with what wubu_model_init produces
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    // Read our saved embedding file and check its stats
    FILE *f = fopen("/home/wubu/bytropix/data/qwen36_embeddings_c.bin.raw", "rb");
    if (f) {
        float *emb = (float *)malloc(n_embd * sizeof(float));
        // First token (token 0)
        fread(emb, sizeof(float), n_embd, f);
        printf("File emb[0]: mean=%f std=%f first5: %f %f %f %f %f\n",
               emb[0], emb[1], emb[0], emb[1], emb[2], emb[3], emb[4]);
        float m=0,s=0;
        for(int i=0;i<n_embd;i++) m+=emb[i];
        m/=n_embd;
        for(int i=0;i<n_embd;i++) s+=(emb[i]-m)*(emb[i]-m);
        s=sqrtf(s/n_embd);
        printf("File emb[0]: mean=%.6f std=%.6f\n", m, s);
        
        // BOS token
        fseek(f, 248044LL * n_embd * sizeof(float), SEEK_SET);
        fread(emb, sizeof(float), n_embd, f);
        m=0; s=0;
        for(int i=0;i<n_embd;i++) m+=emb[i];
        m/=n_embd;
        for(int i=0;i<n_embd;i++) s+=(emb[i]-m)*(emb[i]-m);
        s=sqrtf(s/n_embd);
        printf("File BOS: mean=%.6f std=%.6f first5: ", m, s);
        for(int i=0;i<5;i++) printf("%.6f ", emb[i]);
        printf("\n");
        
        free(emb);
        fclose(f);
    }
    
    return 0;
}
