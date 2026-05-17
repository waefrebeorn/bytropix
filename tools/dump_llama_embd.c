#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *prompt     = argc > 2 ? argv[2] : "Hello";
    int n_gpu_layers = argc > 3 ? atoi(argv[3]) : 0;

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    const llama_vocab *vocab = llama_model_get_vocab(model);

    // Tokenize
    int n_tok = -llama_tokenize(vocab, prompt, strlen(prompt), NULL, 0, true, false);
    std::vector<llama_token> tokens(n_tok);
    if (llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, false) < 0) {
        fprintf(stderr, "Tokenization failed\n"); return 1;
    }
    printf("Tokens: %d\n", n_tok);
    for (int i = 0; i < n_tok; i++) {
        char buf[128];
        int n = llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf), 0, true);
        printf("  [%d] '%.*s'\n", tokens[i], n<0?0:n, n<0?"":buf);
    }

    // Dump token 9419 and token 0 embeddings directly from the model
    int n_embd = llama_model_n_embd(model);
    
    // TODO: llama.cpp doesn't expose raw embeddings easily
    // Let me use the context approach instead
    
    // Create context with embeddings enabled
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024;
    cparams.n_batch = n_tok;
    cparams.embeddings = true;
    cparams.no_perf = true;
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "Failed to decode\n"); return 1;
    }

    // Get embeddings
    float *embd = llama_get_embeddings_seq(ctx, 0);
    if (embd) {
        printf("\nPooled embedding for seq 0:\n");
        for (int i = 0; i < 10; i++) printf(" %.6f", embd[i]);
        printf("\n");
    }

    // Get token embeddings (per-token hidden states)
    for (int i = 0; i < n_tok; i++) {
        float *tok_embd = llama_get_embeddings_ith(ctx, i);
        if (tok_embd) {
            printf("\nToken %d ('%.*s') embedding first 10:\n",
                   tokens[i], 
                   [&]{char b[128];int n=llama_token_to_piece(vocab,tokens[i],b,128,0,true);return n<0?0:n;}(),
                   [&]{char b[128];int n=llama_token_to_piece(vocab,tokens[i],b,128,0,true);return n<0?"":b;}());
            for (int j = 0; j < 10; j++) printf(" %.6f", tok_embd[j]);
            printf("\n");
            float s = 0;
            for (int j = 0; j < n_embd; j++) s += tok_embd[j] * tok_embd[j];
            printf("  rms=%.6f\n", sqrtf(s / n_embd));
            
            // Save to file
            char fname[256];
            snprintf(fname, sizeof(fname), "/tmp/llama_tok%d_embd.bin", tokens[i]);
            FILE *f = fopen(fname, "wb");
            if (f) { fwrite(tok_embd, sizeof(float), n_embd, f); fclose(f); }
            printf("  saved to %s\n", fname);
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
