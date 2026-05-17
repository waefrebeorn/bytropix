/**
 * run_ref_bos.c — Run llama.cpp with BOS-only, dump logits + final hidden.
 * Build: g++ -std=c++11 -O2 -I /home/wubu/llama.cpp/include -I /home/wubu/llama.cpp/ggml/include \
 *   -o run_ref_bos tools/run_ref_bos.c \
 *   -L /home/wubu/llama.cpp/build/bin -lllama -lggml-base -lggml-cpu -lggml \
 *   -lm -Wl,-rpath,/home/wubu/llama.cpp/build/bin
 */
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    ggml_backend_load_all();
    
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    
    const llama_vocab *vocab = llama_model_get_vocab(model);
    int n_embd = llama_model_n_embd(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    
    // BOS token only
    llama_token bos = llama_vocab_bos(vocab);
    std::vector<llama_token> tokens = {bos};
    fprintf(stderr, "BOS token ID: %d (n_embd=%d, n_vocab=%d)\n", bos, n_embd, n_vocab);
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 16;
    cparams.n_batch = 1;
    cparams.embeddings = true;
    
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed ctx\n"); return 1; }
    
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch)) { fprintf(stderr, "Decode failed\n"); return 1; }
    
    // Final hidden state
    const float *embd = llama_get_embeddings_ith(ctx, 0);
    if (embd) {
        FILE *f = fopen("/tmp/ref_hidden_bos.bin", "wb");
        fwrite(embd, sizeof(float), n_embd, f);
        fclose(f);
        fprintf(stderr, "Hidden: rms=%.6f\n", sqrtf([&]{float s=0;for(int i=0;i<n_embd;i++)s+=embd[i]*embd[i];return s/n_embd;}()));
    }
    
    // Logits
    const float *logits = llama_get_logits_ith(ctx, 0);
    if (logits) {
        FILE *f = fopen("/tmp/ref_logits_bos.bin", "wb");
        fwrite(logits, sizeof(float), n_vocab, f);
        fclose(f);
        fprintf(stderr, "Logits dumped (%d)\n", n_vocab);
    }
    
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
