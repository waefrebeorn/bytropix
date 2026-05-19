// Dump llama.cpp's MoE router probabilities and first-layer hidden states for comparison
// Links against libllama.so
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }
    
    ggml_backend_load_all();
    
    auto mparams = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 1;
    cparams.n_ubatch = 1;
    
    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to init context\n"); return 1; }
    
    int n_threads = 16;
    const char * nt_env = getenv("LLAMA_N_THREADS");
    if (nt_env) n_threads = atoi(nt_env);
    llama_set_n_threads(ctx, n_threads, n_threads);
    
    llama_token tokens[] = { (llama_token)248044 };
    auto batch = llama_batch_get_one(tokens, 1);
    
    // Set env to dump hidden states
    // We'll get the embeddings after decode
    
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        return 1;
    }
    
    // Get final logits
    float * logits = llama_get_logits(ctx);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    
    // Save logits
    FILE * f = fopen("/tmp/llama_logits_new.bin", "wb");
    if (f) {
        fwrite(logits, sizeof(float), n_vocab, f);
        fclose(f);
        fprintf(stderr, "Saved logits to /tmp/llama_logits_new.bin\n");
    }
    
    // Also get embeddings (last hidden state before output projection)
    float * embd = llama_get_embeddings(ctx);
    if (embd) {
        int n_embd = llama_model_n_embd(model);
        f = fopen("/tmp/llama_embd_new.bin", "wb");
        if (f) {
            fwrite(embd, sizeof(float), n_embd, f);
            fclose(f);
            fprintf(stderr, "Saved embeddings to /tmp/llama_embd_new.bin\n");
        }
    }
    
    llama_model_free(model);
    return 0;
}
