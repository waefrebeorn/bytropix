// Reference dumper: direct libllama.so linkage for fast per-layer hidden state dumps.
// Replaces llama-cli for generating reference data.
// Usage: DUMP_LAYER_DIR=/tmp/dump_layers ./ref_dumper model.gguf [token_id]
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.gguf [token_id]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    int token_id = 248044;  // default BOS token for Qwen3.6
    if (argc >= 3) token_id = atoi(argv[2]);

    // Load dynamic backends
    ggml_backend_load_all();

    // Model params
    auto mparams = llama_model_default_params();

    // Load model
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model from %s\n", model_path);
        return 1;
    }

    // Vocab
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) {
        fprintf(stderr, "Failed to get vocab\n");
        llama_model_free(model);
        return 1;
    }

    // Context params
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 1;
    cparams.n_ubatch = 1;

    // Init context
    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to init context\n");
        llama_model_free(model);
        return 1;
    }

    // Set threads
    int n_threads = 16;
    const char * nt_env = getenv("LLAMA_N_THREADS");
    if (nt_env) n_threads = atoi(nt_env);
    llama_set_n_threads(ctx, n_threads, n_threads);

    // Tokenize: single token
    llama_token tokens[] = { (llama_token)token_id };
    int n_tokens = 1;

    // Create batch
    auto batch = llama_batch_get_one(tokens, n_tokens);

    // Decode (forward pass). DUMP_LAYER_DIR env var is read by llama-context.cpp
    // at runtime to dump per-layer hidden states.
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        llama_batch_free(batch);
        llama_model_free(model);
        return 1;
    }

    // Get logits
    float * logits = llama_get_logits(ctx);
    int n_vocab = llama_vocab_n_tokens(vocab);

    // Save logits to file (matches existing format)
    const char * out_path = getenv("REF_LOGITS_PATH");
    if (!out_path) out_path = "/tmp/llama_logits_new.bin";
    FILE * f = fopen(out_path, "wb");
    if (f) {
        fwrite(logits, sizeof(float), n_vocab, f);
        fclose(f);
        fprintf(stderr, "Saved %d logits to %s\n", n_vocab, out_path);
    } else {
        fprintf(stderr, "Failed to save logits to %s\n", out_path);
    }

    // Also save hidden state (last token)
    const char * hidden_path = getenv("REF_HIDDEN_PATH");
    if (hidden_path) {
        int n_embd = llama_model_n_embd(model);
        float * embd = llama_get_embeddings(ctx);
        if (embd) {
            FILE * fh = fopen(hidden_path, "wb");
            if (fh) {
                fwrite(embd, sizeof(float), n_embd, fh);
                fclose(fh);
                fprintf(stderr, "Saved %d embeddings to %s\n", n_embd, hidden_path);
            }
        }
    }

    llama_model_free(model);
    // llama_init_from_model ties ctx to model; model_free also frees ctx
    // (batch from llama_batch_get_one is a helper, not heap-allocated)
    
    return 0;
}
