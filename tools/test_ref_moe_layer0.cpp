/**
 * test_ref_moe_layer0.cpp — Use llama.cpp internal API to compute MoE for layer 0 on
 * a simple input (all 1s), dump the MoE output and expert assignments.
 * Build:
 *   g++ -std=c++11 -O2 -I /home/wubu/llama.cpp/include \
 *       -I /home/wubu/llama.cpp/ggml/include -o test_ref_moe_layer0 \
 *       tools/test_ref_moe_layer0.cpp \
 *       -L /home/wubu/llama.cpp/build/bin -lllama -lggml-base -lggml-cpu -lggml \
 *       -lm -Wl,-rpath,/home/wubu/llama.cpp/build/bin
 * Usage: ./test_ref_moe_layer0 model.gguf
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
    if (!model) { fprintf(stderr, "Failed model\n"); return 1; }
    
    const llama_vocab *vocab = llama_model_get_vocab(model);
    int n_embd = llama_model_n_embd(model);
    
    llama_token bos = llama_vocab_bos(vocab);
    std::vector<llama_token> tokens = {bos};
    
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 16;
    cparams.n_batch = tokens.size();
    cparams.embeddings = true;
    
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed ctx\n"); return 1; }
    
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch)) { fprintf(stderr, "Decode failed\n"); return 1; }
    
    // Dump final hidden and logits
    const float *embd = llama_get_embeddings_ith(ctx, 0);
    if (embd) {
        FILE *f = fopen("/tmp/ref_hidden_tok0.bin", "wb");
        fwrite(embd, sizeof(float), n_embd, f);
        fclose(f);
        fprintf(stderr, "Ref hidden: rms=%.6f\n", sqrtf([&](){double s=0;for(int i=0;i<n_embd;i++)s+=embd[i]*embd[i];return s/n_embd;}()));
    }
    
    const float *logits = llama_get_logits_ith(ctx, tokens.size() - 1);
    if (logits) {
        FILE *f = fopen("/tmp/ref_logits_last.bin", "wb");
        fwrite(logits, sizeof(float), llama_vocab_n_tokens(vocab), f);
        fclose(f);
        double s=0; for(int i=0;i<320;i++) s+=logits[i]*logits[i];
        fprintf(stderr, "Ref logits (first 320): rms=%.6f\n", sqrtf(s/320));
    }
    
    // Try to get internal model internals to dump MoE-specific info
    // The llama.cpp API doesn't expose per-layer internals directly,
    // but we can get the final hidden state which already includes MoE.
    // To isolate MoE contribution at layer 0: we need internal access.
    
    fprintf(stderr, "Can't get per-layer MoE output via public API.\n");
    fprintf(stderr, "Need to modify qwen35moe.cpp for per-layer dumps.\n");
    
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
