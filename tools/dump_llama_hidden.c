/**
 * dump_llama_hidden.c — Use llama.cpp API to run model and dump logits.
 * Build: gcc -O2 -I /home/wubu/llama.cpp -I /home/wubu/llama.cpp/ggml/include \
 *        -o dump_llama_hidden tools/dump_llama_hidden.c \
 *        -L /home/wubu/llama.cpp/build/bin -lllama -lggml-base -lggml-cpu \
 *        -lm -Wl,-rpath,/home/wubu/llama.cpp/build/bin
 * Usage: ./dump_llama_hidden model.gguf
 */
#include "llama.h"
#include "common.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }
    
    ggml_backend_load_all();
    
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only
    
    llama_model *model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    
    const llama_vocab *vocab = llama_model_get_vocab(model);
    
    // Tokenize prompt from args (default: empty = BOS only)
    std::string prompt = argc > 2 ? argv[2] : "";
    
    std::vector<llama_token> tokens;
    if (prompt.empty()) {
        llama_token bos = llama_vocab_bos(vocab);
        tokens = {bos};
    } else {
        tokens = common_tokenize(ctx, prompt, true);
    }
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 16;
    cparams.n_batch = 1;
    
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }
    
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "Failed to decode\n"); return 1;
    }
    
    // Get logits for last token
    const float *logits = llama_get_logits_ith(ctx, 0);
    int n_vocab = llama_vocab_n_tokens(vocab);
    
    // Dump full logits
    FILE *f = fopen("/tmp/llama_logits.bin", "wb");
    if (f) { fwrite(logits, sizeof(float), n_vocab, f); fclose(f); }
    printf("Dumped %d logits to /tmp/llama_logits.bin\n", n_vocab);
    
    // Get top-5
    int top5[5] = {0}; float top5v[5] = {-1e30,-1e30,-1e30,-1e30,-1e30};
    for (int i = 0; i < n_vocab; i++) {
        if (logits[i] > top5v[4]) {
            top5v[4] = logits[i]; top5[4] = i;
            for (int k = 3; k >= 0; k--) {
                if (top5v[k] < top5v[k+1]) {
                    float tv = top5v[k]; int ti = top5[k];
                    top5v[k] = top5v[k+1]; top5[k] = top5[k+1];
                    top5v[k+1] = tv; top5[k+1] = ti;
                }
            }
        }
    }
    printf("Top-5 tokens:\n");
    for (int k = 0; k < 5; k++) {
        char buf[256] = {0};
        int n = llama_token_to_piece(vocab, top5[k], buf, sizeof(buf), 0, true);
        printf("  [%d]='%s'(%.2f)\n", top5[k], buf, top5v[k]);
    }
    
    // Also dump the hidden state before output projection (if available)
    // llama.cpp doesn't expose hidden states directly via API
    
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
