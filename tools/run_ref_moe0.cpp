/**
 * run_ref_moe0.c — Run llama.cpp with MOE disabled (pass-through FFN)
 * and dump per-layer residuals + final hidden + logits.
 * Build: g++ -std=c++11 -O2 -I /home/wubu/llama.cpp/include \
 *   -I /home/wubu/llama.cpp/ggml/include -o run_ref_moe0 tools/run_ref_moe0.cpp \
 *   -L /home/wubu/llama.cpp/build/bin -lllama -lggml-base -lggml-cpu -lggml \
 *   -lm -Wl,-rpath,/home/wubu/llama.cpp/build/bin
 *
 * Usage: mkdir -p /tmp/ref_layers && ./run_ref_moe0 model.gguf
 * 
 * NOTE: This does NOT disable MoE — it just runs normal inference and dumps
 * the final hidden state. For layer-by-layer comparison, we need internal access.
 */
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// Allocate a global buffer big enough for the dump
static float g_dump_buf[2048 * 1 * 1];  // [n_embd, n_tokens]
static int g_dump_n = 0;

// Static pointer set by the build function
extern "C" {
    float **llama_dump_post_attn_ptr = NULL;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    ggml_backend_load_all();
    
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed model\n"); return 1; }
    
    const llama_vocab *vocab = llama_model_get_vocab(model);
    int n_embd = llama_model_n_embd(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    
    llama_token bos = llama_vocab_bos(vocab);
    // Support custom token IDs from command line (space-separated)
    std::vector<llama_token> tokens;
    if (argc > 2) {
        // Parse token IDs from argv[2+]
        for (int i = 2; i < argc; i++) {
            tokens.push_back(atoi(argv[i]));
        }
    } else {
        tokens = {bos};
    }
    fprintf(stderr, "Input: %zu tokens\n", tokens.size());
    if (tokens.size() <= 5) {
        fprintf(stderr, "  tokens: ");
        for (auto t : tokens) fprintf(stderr, "%d ", t);
        fprintf(stderr, "\n");
    }
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 16;
    cparams.n_batch = tokens.size();  // Process all tokens in one batch
    cparams.embeddings = true;
    
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed ctx\n"); return 1; }
    
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch)) { fprintf(stderr, "Decode failed\n"); return 1; }
    
    // Final hidden state for each token
    for (int i = 0; i < (int)tokens.size(); i++) {
        const float *embd = llama_get_embeddings_ith(ctx, i);
        if (embd) {
            char fn[256];
            snprintf(fn, sizeof(fn), "/tmp/ref_hidden_tok%d.bin", i);
            FILE *f = fopen(fn, "wb");
            fwrite(embd, sizeof(float), n_embd, f);
            fclose(f);
            double s=0; for(int j=0;j<n_embd;j++) s+=embd[j]*embd[j];
            fprintf(stderr, "Token %d hidden: rms=%.6f\n", i, sqrtf(s/n_embd));
        }
    }
    
    // Logits (last token in batch)
    const float *logits = llama_get_logits_ith(ctx, tokens.size() - 1);
    if (logits) {
        FILE *f = fopen("/tmp/ref_logits_last.bin", "wb");
        fwrite(logits, sizeof(float), n_vocab, f);
        fclose(f);
        fprintf(stderr, "Logits dumped\n");
    }
    
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
