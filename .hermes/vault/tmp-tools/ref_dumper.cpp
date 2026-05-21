// Reference dumper: direct libllama.so linkage for fast per-layer hidden state dumps.
// Replaces llama-cli for generating reference data.
// 
// Usage:
//   # Single token (default token 248044 = BOS)
//   ./ref_dumper model.gguf
//
//   # Single token with intermediate dumps
//   DUMP_LAYER_DIR=/tmp/ref_layers DUMP_INTERMEDIATE_DIR=/tmp/ref_intermediates ./ref_dumper model.gguf
//
//   # Multi-token prompt
//   DUMP_INTERMEDIATE_DIR=/tmp/ref_interm ./ref_dumper model.gguf "The capital of France is" 5
//
//   # With token ID override
//   DUMP_INTERMEDIATE_DIR=/tmp/ref_interm ./ref_dumper model.gguf "" 5 94433

#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.gguf [prompt_string] [n_tokens]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];

    // Load dynamic backends
    ggml_backend_load_all();

    // Model params (CPU only for reference)
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

    // Get token count from args
    int n_tokens_to_gen = 0; // 0 = prefill only, no generation
    std::string prompt_str;
    bool raw_token_mode = false;
    int token_id_override = 248044; // default BOS for Qwen3.6
    if (argc >= 3 && argv[2][0] != '\0' && argv[2][0] != '-') {
        // Check if argv[2] is a numeric token ID
        bool is_numeric = true;
        for (const char *p = argv[2]; *p; p++) {
            if (*p < '0' || *p > '9') { is_numeric = false; break; }
        }
        if (is_numeric) {
            raw_token_mode = true;
            token_id_override = atoi(argv[2]);
        } else {
            prompt_str = argv[2];
        }
    }
    if (argc >= 4) {
        n_tokens_to_gen = atoi(argv[3]);
    }

    // Tokenize prompt if provided, otherwise use default token
    std::vector<llama_token> tokens;
    if (!prompt_str.empty()) {
        int n_tok = -llama_tokenize(vocab, prompt_str.c_str(), prompt_str.length(), NULL, 0, true, false);
        if (n_tok <= 0) {
            fprintf(stderr, "Failed to tokenize prompt\n");
            llama_model_free(model);
            return 1;
        }
        tokens.resize(n_tok);
        llama_tokenize(vocab, prompt_str.c_str(), prompt_str.length(), tokens.data(), tokens.size(), true, false);
        fprintf(stderr, "Tokenized prompt to %d tokens\n", n_tok);
    } else if (raw_token_mode) {
        tokens.push_back((llama_token)token_id_override);
        fprintf(stderr, "Using raw token ID %d\n", token_id_override);
    } else {
        tokens.push_back((llama_token)token_id_override);
        fprintf(stderr, "Using default BOS token ID %d\n", token_id_override);
    }

    // Context params
    int n_ctx = (int)tokens.size() + n_tokens_to_gen + 64;
    if (n_ctx < 512) n_ctx = 512;

    auto cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = (int)tokens.size() > 32 ? (int)tokens.size() : 32;
    cparams.n_ubatch = (int)tokens.size() > 32 ? (int)tokens.size() : 32;

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

    // Prefill batch
    auto batch = llama_batch_get_one(tokens.data(), (int)tokens.size());

    fprintf(stderr, "Prefilling %zu tokens...\n", tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode (prefill) failed\n");
        llama_model_free(model);
        return 1;
    }
    fprintf(stderr, "Preflill done. GPU VRAM info not available.\n");

    // Generate additional tokens one by one (each decode triggers dump)
    std::vector<llama_token> generated;
    for (int i = 0; i < n_tokens_to_gen; i++) {
        float * logits = llama_get_logits(ctx);
        int n_vocab = llama_vocab_n_tokens(vocab);

        // Argmax
        int best = 0;
        float best_val = logits[0];
        for (int j = 1; j < n_vocab; j++) {
            if (logits[j] > best_val) {
                best_val = logits[j];
                best = j;
            }
        }
        generated.push_back((llama_token)best);

        // Decode this token
        auto gen_batch = llama_batch_get_one(&generated.back(), 1);
        if (llama_decode(ctx, gen_batch) != 0) {
            fprintf(stderr, "llama_decode (gen %d) failed\n", i);
            break;
        }
    }

    // Get final logits
    float * logits = llama_get_logits(ctx);
    int n_vocab = llama_vocab_n_tokens(vocab);

    // Save logits to file
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

    // Show generated tokens
    if (!generated.empty()) {
        fprintf(stderr, "Generated %zu tokens: [", generated.size());
        for (size_t i = 0; i < generated.size() && i < 5; i++) {
            fprintf(stderr, "%s%d", i > 0 ? " " : "", generated[i]);
        }
        if (generated.size() > 5) fprintf(stderr, " ...");
        fprintf(stderr, "]\n");

        // Detokenize
        char buf[1024];
        int len = llama_token_to_piece(vocab, generated[0], buf, sizeof(buf), 0, false);
        if (len > 0) {
            buf[len] = '\0';
            int max_show = (int)generated.size() < 4 ? (int)generated.size() : 4;
            fprintf(stderr, "Generated text fragment: ");
            for (int j = 0; j < max_show; j++) {
                len = llama_token_to_piece(vocab, generated[j], buf, sizeof(buf), 0, false);
                if (len > 0) {
                    buf[len] = '\0';
                    fprintf(stderr, "%s", buf);
                }
            }
            fprintf(stderr, "\n");
        }
    }

    llama_model_free(model);
    // llama_init_from_model ties ctx to model; model_free also frees ctx
    
    return 0;
}
