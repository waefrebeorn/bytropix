/**
 * dump_llama_logits.c — Use llama.cpp API to dump logits for a single BOS token.
 * Build: g++ -std=c++11 -O2 -I /home/wubu/llama.cpp/include -I /home/wubu/llama.cpp/ggml/include \
 *        -o dump_llama_logits tools/dump_llama_logits.c \
 *        -L /home/wubu/llama.cpp/build/bin -lllama -lggml-base -lggml-cpu -lggml \
 *        -lm -Wl,-rpath,/home/wubu/llama.cpp/build/bin
 * Usage: ./dump_llama_logits model.gguf [output_path]
 */
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf [output_path]\n", argv[0]); return 1; }
    const char *outpath = argc > 2 ? argv[2] : "/tmp/llama_logits.bin";

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap = false;

    llama_model *model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 16;
    cparams.n_batch = 16;
    cparams.embeddings = true;  // enable embedding (hidden state) output

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    // Tokenize prompt
    const char *prompt = argc > 3 ? argv[3] : "";
    std::string prompt_str = prompt;
    
    int n_tokens;
    if (prompt_str.empty()) {
        // BOS only
        llama_token bos = llama_vocab_bos(vocab);
        n_tokens = 1;
        std::vector<llama_token> tokens = {bos};
        llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "Failed to decode\n"); return 1;
        }
    } else {
        // Tokenize
        int n_prompt = -llama_tokenize(vocab, prompt, prompt_str.size(), NULL, 0, true, true);
        std::vector<llama_token> tokens(n_prompt);
        if (llama_tokenize(vocab, prompt, prompt_str.size(), tokens.data(), tokens.size(), true, true) < 0) {
            fprintf(stderr, "Failed to tokenize\n"); return 1;
        }
        n_tokens = n_prompt;
        llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "Failed to decode\n"); return 1;
        }
    }

    // Also dump the final hidden state (embeddings) for comparison
    const float *embd = llama_get_embeddings_ith(ctx, n_tokens - 1);
    int n_embd = llama_model_n_embd(model);
    fprintf(stderr, "n_embd = %d\n", n_embd);

    if (embd) {
        // Determine hidden path: replace .bin with _hidden.bin
        std::string hidden_path = outpath;
        std::string::size_type dot = hidden_path.rfind(".bin");
        if (dot != std::string::npos) {
            hidden_path.replace(dot, 4, "_hidden.bin");
        } else {
            hidden_path += "_hidden";
        }

        {
            FILE *fh = fopen(hidden_path.c_str(), "wb");
            if (fh) { fwrite(embd, sizeof(float), n_embd, fh); fclose(fh); }
        }
        fprintf(stderr, "Dumped %d hidden states to %s\n", n_embd, hidden_path.c_str());

        // Compare embeddings stats
        { float mn=1e30,mx=-1e30,sum=0,sumsq=0;
          for(int i=0;i<n_embd;i++){if(embd[i]<mn)mn=embd[i];if(embd[i]>mx)mx=embd[i];sum+=embd[i];sumsq+=embd[i]*embd[i];}
          fprintf(stderr, "Hidden: mean=%.4f rms=%.4f range=[%.4f,%.4f]\n", sum/n_embd, sqrtf(sumsq/n_embd), mn, mx);
        }
    } else {
        fprintf(stderr, "Warning: embeddings not available\n");
    }

    // Get logits for last token
    const float *logits = llama_get_logits_ith(ctx, n_tokens - 1);
    
    FILE *f = fopen(outpath, "wb");
    if (f) { fwrite(logits, sizeof(float), n_vocab, f); fclose(f); }
    fprintf(stderr, "Dumped %d logits to %s\n", n_vocab, outpath);

    // Top-5
    float top5v[5] = {-1e30,-1e30,-1e30,-1e30,-1e30};
    int top5[5] = {0};
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
    fprintf(stderr, "Top-5 tokens:\n");
    for (int k = 0; k < 5; k++) {
        char buf[256] = {0};
        int n = llama_token_to_piece(vocab, top5[k], buf, sizeof(buf), 0, true);
        if (n < 0) { buf[0] = '?'; buf[1] = 0; }
        fprintf(stderr, "  [%d]='%s'(%.2f)\n", top5[k], buf, top5v[k]);
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
