#include "wubu_gemma4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Simple argmax sampling
static int sample_argmax(const float *logits, int n_vocab) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [prompt] [n_tokens]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *prompt = argc > 2 ? argv[2] : "The capital of France is";
    int n_tokens = argc > 3 ? atoi(argv[3]) : 20;

    // Load model
    gemma4_model_t model;
    if (!gemma4_init(&model, model_path)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Init KV cache
    gemma4_kv_cache_t cache;
    if (!gemma4_kv_cache_init(&cache, 4096)) {
        fprintf(stderr, "Failed to init KV cache\n");
        gemma4_free(&model);
        return 1;
    }

    printf("\nPrompt: %s\n", prompt);
    printf("Generating %d tokens...\n\n", n_tokens);
    fflush(stdout);

    // Simple tokenization: use token IDs directly for now
    // For real use, we'd load the tokenizer from GGUF
    // For now, use a known token: "The" = 4695 in Gemma 4 tokenizer
    // Actually, let's just start with token 0 and see what we get
    // Better approach: use a fixed input to test
    
    // Gemma 4 BOS token = 2
    // "The" ≈ token around that range... let's just test with token 2 as BOS
    
    // Actually, for a proper test, let me prefill with a few known tokens
    // Gemma 4 uses SentencePiece. "The" might be around 4695
    // Let me just test with token 2 (BOS) and see logits output

    printf("Testing with BOS token...\n");

    float *logits = (float *)malloc(GEMMA4_VOCAB_SIZE * sizeof(float));

    // Forward BOS token
    gemma4_forward(&model, &cache, 2, 0, logits);

    int first_token = sample_argmax(logits, GEMMA4_VOCAB_SIZE);
    printf("BOS -> token %d (logit=%.4f)\n", first_token, logits[first_token]);

    // Generate more tokens
    int token = first_token;
    printf("\nGenerated: ");
    for (int i = 0; i < n_tokens; i++) {
        gemma4_forward(&model, &cache, token, i + 1, logits);
        token = sample_argmax(logits, GEMMA4_VOCAB_SIZE);
        printf("%d ", token);
        fflush(stdout);
    }
    printf("\n");

    free(logits);
    gemma4_kv_cache_free(&cache);
    gemma4_free(&model);

    return 0;
}
