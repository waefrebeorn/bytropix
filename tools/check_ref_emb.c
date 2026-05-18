/* Extract token embedding from llama.cpp for comparison */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "llama.h"

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    llama_backend_init();
    
    struct llama_model_params lmp = llama_model_default_params();
    lmp.n_gpu_layers = 0;
    lmp.use_mmap = false;  // needed to access raw tensors? Let's use the model normally
    struct llama_model *model = llama_model_load_from_file(path, lmp);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int n_embd = llama_model_n_embd(model);
    
    struct llama_context_params lcp = llama_context_default_params();
    lcp.n_ctx = 128;
    struct llama_context *ctx = llama_init_from_model(model, lcp);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }
    
    llama_token bos = llama_vocab_bos(vocab);
    printf("BOS token = %d, n_embd = %d\n", bos, n_embd);
    
    // Check if llama embeds can be accessed
    // Actually for llama.cpp, we need to decode and then look at internal state
    
    // Let's use embedding mode: has_embd=true
    // Or we can access the raw model tensors via internal API...
    // There's no public API for token embedding in llama.h
    
    // Alternative: save output logit for the first token and compare to our
    // embedding-derived logit
    
    // Run BOS token
    llama_token tokens[1] = {bos};
    struct llama_batch batch = llama_batch_get_one(tokens, 1);
    if (llama_decode(ctx, batch)) { fprintf(stderr, "Decode failed\n"); return 1; }
    
    // Get the logits
    float *logits = llama_get_logits_ith(ctx, 0);
    
    // Now let's manually compute what 0-layer logits should look like
    // We need the token embedding after RMSNorm and output projection
    
    // Actually let's just compare our 40-layer logits against ref
    // Since the reference IS the 40-layer output, it should be the same
    
    // But what I really need is the initial token embedding.
    // llama.cpp doesn't expose this publicly.
    // The only way is to look at the files or use a patched llama.cpp
    
    // Let's try triton: get llama internal memview ?
    
    // For now, let's dump everything we can
    printf("Ref logits first 10: ");
    for (int i = 0; i < 10 && i < n_embd; i++) printf("%.4f ", logits[i]);
    printf("\n");
    
    // Also get last hidden state
    float *embd = NULL;
    
#if LLAMA_API_VERSION_MINOR >= 3
    // llama_get_embeddings_ith exists in newer versions
    embd = llama_get_embeddings_ith(ctx, 0);
    if (embd) {
        printf("Last hidden first 10: ");
        for (int i = 0; i < 10; i++) printf("%.4f ", embd[i]);
        printf("\n");
    } else {
        printf("No embeddings available\n");
    }
#endif
    
    float *ref = (float*)malloc(n_embd * sizeof(float));
    memcpy(ref, logits, n_embd * sizeof(float));
    
    // Save
    FILE *f = fopen("/tmp/ref_logits.bin", "wb");
    fwrite(ref, sizeof(float), llama_model_n_vocab(model), f);
    fclose(f);
    
    if (embd) {
        f = fopen("/tmp/ref_hidden.bin", "wb");
        fwrite(embd, sizeof(float), n_embd, f);
        fclose(f);
        printf("Saved ref hidden to /tmp/ref_hidden.bin\n");
    }
    
    printf("Saved ref logits (%d dims) to /tmp/ref_logits.bin\n", llama_model_n_vocab(model));
    
    // Now let's try to directly read the embedding from the GGUF using llama's API
    // First check if we can access the embedding tensor
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
