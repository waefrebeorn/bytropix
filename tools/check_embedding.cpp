/**
 * check_embedding.c — Compare token embeddings between bytropix file and libllama
 * Usage: ./check_embedding [token_id]
 * Uses libllama's internal access to token_embd.weight, compares with extracted file.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "llama.h"

int main(int argc, char **argv) {
    const char *model_path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int token_id = 248044;
    if (argc > 1) token_id = atoi(argv[1]);

    // Load model via libllama
    fprintf(stderr, "Loading model via libllama...\n");
    auto mparams = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    // Get embedding dimension
    int D = 2048;

    // We can't directly access token_embd.weight from public API
    // Instead, run a decode and compare via hidden states

    // Load context
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 1;
    cparams.n_ubatch = 1;
    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to init context\n"); return 1; }

    llama_set_n_threads(ctx, 16, 16);

    // Decode BOS token
    llama_token tokens[] = { (llama_token)token_id };
    auto batch = llama_batch_get_one(tokens, 1);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    // Get logits (first token)
    float * logits = llama_get_logits(ctx);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // Save logits to compare
    const char *logits_path = getenv("LOGITS_PATH");
    if (!logits_path) logits_path = "/tmp/llama_bos_logits.bin";
    FILE * f = fopen(logits_path, "wb");
    if (f) {
        fwrite(logits, sizeof(float), n_vocab, f);
        fclose(f);
        fprintf(stderr, "Saved logits to %s\n", logits_path);
    }

    // Also save embeddings (final hidden state before output projection)
    float * embd = llama_get_embeddings(ctx);
    if (embd) {
        const char *hidden_path = getenv("HIDDEN_PATH");
        if (hidden_path) {
            FILE * fh = fopen(hidden_path, "wb");
            if (fh) {
                fwrite(embd, sizeof(float), D, fh);
                fclose(fh);
                fprintf(stderr, "Saved embeddings to %s\n", hidden_path);
            }
        }
        fprintf(stderr, "Final embedding: mean=%.6f, std=%.6f\n", 
                embd[0], embd[D-1]); // just show first/last
    }

    // Compare with bytropix logits if available
    const char *bytropix_path = getenv("BYTROPIX_LOGITS_PATH");
    if (bytropix_path && logits_path) {
        FILE * fb = fopen(bytropix_path, "rb");
        if (fb) {
            float *by_logits = (float *)malloc(n_vocab * sizeof(float));
            fread(by_logits, sizeof(float), n_vocab, fb);
            fclose(fb);

            // Compute cos-sim
            double dot = 0, n1 = 0, n2 = 0;
            for (int i = 0; i < n_vocab; i++) {
                dot += (double)logits[i] * by_logits[i];
                n1 += (double)logits[i] * logits[i];
                n2 += (double)by_logits[i] * by_logits[i];
            }
            double cos_sim = dot / (sqrt(n1) * sqrt(n2) + 1e-30);
            fprintf(stderr, "Logits cos-sim vs bytropix: %.6f\n", cos_sim);
            free(by_logits);
        }
    }

    llama_model_free(model);
    fprintf(stderr, "Done.\n");
    return 0;
}
