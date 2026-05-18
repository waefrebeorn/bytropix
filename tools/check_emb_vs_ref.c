/* Dump the actual token embedding for BOS from llama.cpp model tensors */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "llama.h"

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    llama_backend_init();
    
    struct llama_model_params lmp = llama_model_default_params();
    lmp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(path, lmp);
    if (!model) return 1;
    
    int n_embd = llama_model_n_embd(model);
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    printf("n_embd=%d n_vocab=%d\n", n_embd, n_vocab);
    
    // Decode BOS token with embedding extraction
    struct llama_context_params lcp = llama_context_default_params();
    lcp.n_ctx = 128;
    lcp.embeddings = true;  // Enable hidden state output
    struct llama_context *ctx = llama_init_from_model(model, lcp);
    
    // We want the embedding for token 0 (first token, not BOS)
    llama_token tok0 = 0;  // token index 0
    struct llama_batch batch = llama_batch_get_one(&tok0, 1);
    
    if (llama_decode(ctx, batch)) {
        printf("Decode failed\n");
    }
    
    // Get the hidden state (this is AFTER all layers, not the initial embedding)
    float *hidden = llama_get_embeddings_ith(ctx, 0);
    if (hidden) {
        printf("Ref hidden (post-40L) for tok0: ");
        for (int i = 0; i < 5; i++) printf("%.4f ", hidden[i]);
        printf("\n");
    }
    
    // Alternative: use a model that only has the embedding layer
    // Actually, we can't easily get just the embedding from llama.cpp API
    
    // Let's instead just compare our saved embedding by its distribution
    // Check if token 0 embedding is reasonable
    
    // Read our file
    FILE *f = fopen("/home/wubu/bytropix/data/qwen36_embeddings_c.bin.raw", "rb");
    if (!f) { printf("Cannot open embedding file\n"); return 1; }
    
    float *crnt = (float *)malloc(n_embd * sizeof(float));
    
    // Read first 5 tokens
    for (int t = 0; t < 5; t++) {
        fread(crnt, sizeof(float), n_embd, f);
        float m=0,s=0;
        for(int i=0;i<n_embd;i++){m+=crnt[i];s+=crnt[i]*crnt[i];}
        m/=n_embd;
        s=s/n_embd - m*m;
        s=sqrtf(fabsf(s));
        printf("File tok[%d]: mean=%.6f std=%.6f first=%.6f\n", t, m, s, crnt[0]);
    }
    
    // Read BOS token (248044)
    fseek(f, 248044LL * n_embd * sizeof(float), SEEK_SET);
    fread(crnt, sizeof(float), n_embd, f);
    float m=0,s=0;
    for(int i=0;i<n_embd;i++){m+=crnt[i];s+=crnt[i]*crnt[i];}
    m/=n_embd;
    s=s/n_embd - m*m;
    s=sqrtf(fabsf(s));
    printf("File BOS: mean=%.6f std=%.6f first5=", m, s);
    for(int i=0;i<5;i++) printf("%.6f ", crnt[i]);
    printf("\n");
    
    free(crnt);
    fclose(f);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
