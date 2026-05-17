/**
 * dump_ref_layers.c — Use llama.cpp to dump per-layer hidden states.
 * Build:
 * g++ -std=c++11 -O2 -I /home/wubu/llama.cpp/include -I /home/wubu/llama.cpp/ggml/include \
 *     -o dump_ref_layers tools/dump_ref_layers.c \
 *     -L /home/wubu/llama.cpp/build/bin -lllama -lggml-base -lggml-cpu -lggml \
 *     -lm -Wl,-rpath,/home/wubu/llama.cpp/build/bin
 */
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// Callback to capture per-layer hidden states
struct layer_capture {
    float *data[256]; // max 256 layers
    int n_layers;
    int n_embd;
};

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model *model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    int n_layer = llama_model_n_layer(model);
    int n_embd = llama_model_n_embd(model);
    fprintf(stderr, "Model: %d layers, %d embd\n", n_layer, n_embd);

    // We'll use a new enough API: register layer callback
    // For now, use a simpler approach: get final hidden state
    // and compare against ours
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 16;
    cparams.n_batch = 1;
    cparams.embeddings = true;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    // BOS only
    const llama_vocab *vocab = llama_model_get_vocab(model);
    llama_token bos = llama_vocab_bos(vocab);
    std::vector<llama_token> tokens = {bos};

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "Failed to decode\n"); return 1;
    }

    // Get final hidden state  
    const float *embd = llama_get_embeddings_ith(ctx, 0);
    if (embd) {
        FILE *f = fopen("/tmp/ref_post_moe_final.bin", "wb");
        if (f) { fwrite(embd, sizeof(float), n_embd, f); fclose(f); }
        fprintf(stderr, "Dumped final hidden (%d dims)\n", n_embd);
    }

    // Get logits
    const float *logits = llama_get_logits_ith(ctx, 0);
    int n_vocab = llama_vocab_n_tokens(vocab);
    FILE *fl = fopen("/tmp/ref_logits_bos.bin", "wb");
    if (fl) { fwrite(logits, sizeof(float), n_vocab, fl); fclose(fl); }
    fprintf(stderr, "Dumped logits (%d vocab)\n", n_vocab);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
