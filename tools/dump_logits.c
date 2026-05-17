/**
 * dump_logits.c — Dump logits from llama.cpp for a given prompt.
 * Build:
 *   g++ -O2 -I /home/wubu/llama.cpp/include -I /home/wubu/llama.cpp/ggml/include \
 *       -o dump_logits dump_logits.c \
 *       -L/home/wubu/llama.cpp/build/bin -Wl,-rpath,/home/wubu/llama.cpp/build/bin \
 *       -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda -lm -lpthread -lstdc++
 */
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    const char *model_path  = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *prompt      = argc > 2 ? argv[2] : "Hello";
    const char *logits_out  = argc > 3 ? argv[3] : "/tmp/ref_logits.bin";
    const char *embd_out    = argc > 4 ? argv[4] : "/tmp/ref_embd.bin";
    int n_gpu_layers = argc > 5 ? atoi(argv[5]) : 0;

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

    printf("Tokens: %d\n  ", n_tok);
    for (int i = 0; i < n_tok; i++) {
        char buf[128];
        int n = llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf), 0, true);
        if (n < 0) n = 0;
        printf("[%d]'%.*s' ", tokens[i], n, buf);
    }
    printf("\n");

    // Context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024;
    cparams.n_batch = n_tok;
    cparams.embeddings = true;
    cparams.no_perf = true;
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    // Batch + decode
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "Failed to decode\n"); return 1;
    }

    // Get logits for last token
    int n_vocab = llama_vocab_n_tokens(vocab);
    float *logits = llama_get_logits_ith(ctx, -1);
    printf("Logits: %d vocab\n  first 5: %.6f %.6f %.6f %.6f %.6f\n",
           n_vocab, logits[0], logits[1], logits[2], logits[3], logits[4]);

    FILE *f = fopen(logits_out, "wb");
    if (f) { fwrite(logits, sizeof(float), n_vocab, f); fclose(f); }
    printf("Wrote logits to %s (%d bytes)\n", logits_out, n_vocab * 4);

    // Get embeddings
    int n_embd = llama_model_n_embd(model);
    float *embd = llama_get_embeddings_seq(ctx, 0);
    if (embd) {
        printf("Embeddings (seq 0): %d dims\n  first 5: %.6f %.6f %.6f %.6f %.6f\n",
               n_embd, embd[0], embd[1], embd[2], embd[3], embd[4]);
        f = fopen(embd_out, "wb");
        if (f) { fwrite(embd, sizeof(float), n_embd, f); fclose(f); }
        printf("Wrote embeddings to %s (%d bytes)\n", embd_out, n_embd * 4);
    } else {
        printf("No pooled embeddings. Trying token embeddings...\n");
        float *tok_embd = llama_get_embeddings_ith(ctx, -1);
        if (tok_embd) {
            printf("Token embedding (-1): %d dims\n  first 5: %.6f %.6f %.6f %.6f %.6f\n",
                   n_embd, tok_embd[0], tok_embd[1], tok_embd[2], tok_embd[3], tok_embd[4]);
            f = fopen(embd_out, "wb");
            if (f) { fwrite(tok_embd, sizeof(float), n_embd, f); fclose(f); }
            printf("Wrote token embedding to %s\n", embd_out);
        } else {
            fprintf(stderr, "No embeddings available at all\n");
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
