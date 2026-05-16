/**
 * ref_forward.c — Use libllama to run a single token forward, dump hidden state.
 * Links against ~/llama.cpp/build/bin/libllama.so
 *
 * Build:
 *   gcc -O2 -I/home/wubu/llama.cpp/include -I/home/wubu/llama.cpp/ggml/include \
 *       -o ref_forward tools/ref_forward.c \
 *       -L/home/wubu/llama.cpp/build/bin -Wl,-rpath,/home/wubu/llama.cpp/build/bin \
 *       -lllama -lggml-base -lggml-cpu -lggml-cuda \
 *       -lm -lstdc++ -lssl -lcrypto
 *
 * Usage: ./ref_forward model.gguf "Hello"
 */
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s model.gguf prompt\n", argv[0]); return 1; }

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;

    struct llama_model *model = llama_model_load_from_file(argv[1], model_params);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    ctx_params.embeddings = true;

    struct llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    int n_layers = llama_model_n_layer(model);
    int n_embd = llama_model_n_embd(model);
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    printf("{\"n_layers\": %d, \"n_embd\": %d, \"n_vocab\": %d}\n", n_layers, n_embd, n_vocab);

    // Tokenize without automatic special tokens, then manually add BOS
    const char *prompt = argv[2];
    int n_tokens = strlen(prompt) + 3;
    llama_token *tokens = (llama_token *)malloc(n_tokens * sizeof(llama_token));
    int n_tokenized = llama_tokenize(vocab, prompt, strlen(prompt), tokens + 1, n_tokens - 1, false, false);
    tokens[0] = 248044; // manual BOS
    n_tokens = 1 + n_tokenized;
    printf("{\"n_tokens\": %d}\\n", n_tokens);
    printf("{\"token_ids\": [");
    for (int i = 0; i < n_tokens; i++) {
        if (i > 0) printf(", ");
        printf("%d", tokens[i]);
    }
    printf("]}\n");

    // Run forward pass
    llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n"); return 1;
    }

    // Try to get embeddings (hidden states BEFORE last layer norm + output proj)
    float *embeddings = llama_get_embeddings(ctx);
    if (!embeddings) {
        fprintf(stderr, "No embeddings returned (did you set embeddings=true?)\n"); return 1;
    }

    // DUMP_EMB: write initial token embedding (first token's raw embedding)
    {
        const char *de = getenv("DUMP_EMB");
        if (de) {
            // Get the embedding for first token via a separate approach
            // llama_get_embeddings gives output, not input. Use a different approach
        }
    }

    // Get final (last token) hidden state
    if (embeddings) {
        float *last = embeddings + (n_tokens - 1) * n_embd;
        float mean = 0, maxv = -1e30, minv = 1e30;
        for (int i = 0; i < n_embd; i++) {
            mean += last[i];
            if (last[i] > maxv) maxv = last[i];
            if (last[i] < minv) minv = last[i];
        }
        mean /= n_embd;
        printf("{\"final_hidden_mean\": %.6f, \"final_hidden_max\": %.6f, \"final_hidden_min\": %.6f}\n", mean, maxv, minv);
        printf("{\"final_hidden[:10]\": [");
        for (int i = 0; i < 10; i++) {
            if (i > 0) printf(", ");
            printf("%.8f", last[i]);
        }
        printf("]}\n");

        // Dump full final hidden state to binary file
        FILE *f = fopen("/tmp/ref_last_hidden.bin", "wb");
        if (f) {
            fwrite(last, sizeof(float), n_embd, f);
            fclose(f);
            printf("{\"dump\": \"/tmp/ref_last_hidden.bin\"}\n");
        }
    }

    // Get logits
    float *logits = llama_get_logits_ith(ctx, n_tokens - 1);
    if (logits) {
        int top_idx = 0; float top_val = -1e30;
        for (int i = 0; i < n_vocab; i++) {
            if (logits[i] > top_val) { top_val = logits[i]; top_idx = i; }
        }
        printf("{\"top_token\": %d, \"top_value\": %.4f}\n", top_idx, top_val);

        // Dump logits
        FILE *f = fopen("/tmp/ref_logits.bin", "wb");
        if (f) {
            fwrite(logits, sizeof(float), n_vocab, f);
            fclose(f);
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    free(tokens);
    return 0;
}
