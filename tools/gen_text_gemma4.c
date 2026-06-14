#include "wubu_gemma4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/prctl.h>

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
    // Disable core dumps to avoid 16GB+ crash files
    {
        struct rlimit rl = {0, 0};
        setrlimit(RLIMIT_CORE, &rl);
        prctl(PR_SET_DUMPABLE, 0);
    }

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [prompt] [n_tokens]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    int n_tokens = argc > 3 ? atoi(argv[3]) : 20;

    // Load model
    g4_model_t model;
    if (!g4_model_init(&model, model_path)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    printf("\nGenerating %d tokens...\n\n", n_tokens);
    fflush(stdout);

    // Gemma 4 BOS token = 2
    float *logits = (float *)malloc((size_t)G4_VOCAB * sizeof(float));
    if (!logits) { fprintf(stderr, "OOM\n"); g4_model_destroy(&model); return 1; }

    // Forward BOS token
    g4_model_decode(&model, 2, logits);

    int token = sample_argmax(logits, G4_VOCAB);
    printf("BOS -> token %d (logit=%.4f)\n", token, logits[token]);

    // Generate more tokens
    printf("\nGenerated: ");
    for (int i = 0; i < n_tokens; i++) {
        g4_model_decode(&model, token, logits);
        token = sample_argmax(logits, G4_VOCAB);
        printf("%d ", token);
        fflush(stdout);
    }
    printf("\n");

    free(logits);
    g4_model_destroy(&model);

    return 0;
}
