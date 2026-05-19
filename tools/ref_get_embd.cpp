#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    
    auto mparams = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", mparams);
    
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    struct llama_context * ctx = llama_init_from_model(model, cparams);
    
    llama_set_n_threads(ctx, 16, 16);
    
    llama_token tokens[] = { (llama_token)248044 };
    auto batch = llama_batch_get_one(tokens, 1);
    
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    
    // Get embeddings (last hidden state before output proj)
    float * embd = llama_get_embeddings(ctx);
    int n_embd = llama_model_n_embd(model);
    
    if (embd) {
        FILE * f = fopen("/tmp/ref_final_hidden.bin", "wb");
        fwrite(embd, sizeof(float), n_embd, f);
        fclose(f);
        fprintf(stderr, "Saved %d embeddings\n", n_embd);
    }
    
    // Get logits
    float * logits = llama_get_logits(ctx);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * f = fopen("/tmp/llama_logits_new.bin", "wb");
    fwrite(logits, sizeof(float), n_vocab, f);
    fclose(f);
    fprintf(stderr, "Saved logits\n");
    
    llama_model_free(model);
    return 0;
}
