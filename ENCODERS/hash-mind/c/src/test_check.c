#include "tokenizer.h"
#include "hashmind_model.h"
#include "rolling_hash.h"
#include <stdio.h>
#include <string.h>

int main() {
    Tokenizer tok;
    tokenizer_init(&tok);
    
    HashMindModel model;
    hashmind_model_init(&model);
    
    const char* text = "abcdef";
    int tokens[16];
    int n = tokenizer_encode(&tok, text, tokens, 16);
    
    uint32_t hashes[16];
    int nh;
    rolling_hash_all(tokens, n, 3, hashes, &nh);
    
    printf("Tokens (%d):", n);
    for (int i = 0; i < n; i++) printf(" %d", tokens[i]);
    printf("\nHashes (%d):", nh);
    for (int i = 0; i < nh; i++) printf(" %u", hashes[i]);
    printf("\n\n");
    
    /* Single forward + loss check */
    float logits[VOCAB_SIZE];
    BlockActs acts;
    hashmind_forward(&model, hashes, nh, tokens, n, logits, &acts);
    
    printf("Logits:");
    for (int i = 0; i < 10; i++) printf(" %.4f", logits[i]);
    printf(" ...\n");
    
    float loss = nn_cross_entropy_loss(logits, tokens[1], VOCAB_SIZE);
    printf("Loss: %.6f\n", loss);
    
    /* Test gradient */
    float dlogits[VOCAB_SIZE];
    nn_cross_entropy_grad(logits, tokens[1], VOCAB_SIZE, dlogits);
    printf("dlogits:");
    for (int i = 0; i < 5; i++) printf(" %.6f", dlogits[i]);
    printf(" ...\n");
    
    /* Check for nan */
    int has_nan = 0;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (isnan(logits[i]) || isinf(logits[i])) {
            printf("  logits[%d] = %.4f IS NAN/INF!\n", i, logits[i]);
            has_nan = 1;
        }
    }
    if (!has_nan) printf("No NaN in logits - forward pass OK\n");
    if (isnan(loss) || isinf(loss)) printf("Loss is NAN/INF!\n");
    
    /* Test gradient apply counts */
    printf("\nParam count: %ld\n", hashmind_param_count());
    printf("Model size: %zu bytes\n", sizeof(HashMindModel));
    printf("Grad size: %zu bytes\n", sizeof(HashMindGrad));
    
    return 0;
}
