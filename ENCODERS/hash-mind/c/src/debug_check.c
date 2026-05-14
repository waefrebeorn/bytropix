#include "tokenizer.h"
#include "hashmind_model.h"
#include "hashmind_data.h"
#include <stdio.h>
#include <string.h>

int main() {
    Tokenizer tok;
    tokenizer_init(&tok);

    HashMindModel model;
    HashMindGrad grad;
    HashMindMomentum vel;
    hashmind_model_init(&model);
    hashmind_zero_grad(&grad);
    memset(&vel, 0, sizeof(vel));

    TrainCtx ctx = {.lr = 0.001f, .momentum = 0.9f, .weight_decay = 1e-4f, .step = 0, .model = &model, .grad = &grad, .vel = &vel};

    const char* data = "hello world this is a test of the c training system.";
    TextData td;
    textdata_init(&td, &tok, data);

    printf("=== Debug Train Step 1 ===\n");
    TrainExample ex;
    textdata_next(&td, &ex);

    float logits[VOCAB_SIZE];
    BlockActs acts;
    hashmind_forward(&model, ex.input_hashes, CONTEXT_LEN,
                     ex.input_tokens, CONTEXT_LEN, logits, &acts);

    float loss = nn_cross_entropy_loss(logits, ex.target_token, VOCAB_SIZE);
    printf("Step 1 loss: %.6f\n", loss);
    printf("Target: %d (char '%c')\n", ex.target_token, tok.idx_to_char[ex.target_token]);
    printf("Logit[target]: %.4f\n", logits[ex.target_token]);
    
    /* Check for nan in logits */
    for (int i = 0; i < VOCAB_SIZE; i++) if (isnan(logits[i])) { printf("NaN at logits[%d]\n", i); break; }

    float dlogits[VOCAB_SIZE];
    nn_cross_entropy_grad(logits, ex.target_token, VOCAB_SIZE, dlogits);
    printf("dlogits sum: %.6f (should be ~0)\n", dlogits[0]+dlogits[1]);

    hashmind_backward(&model, &grad, dlogits, ex.input_hashes, CONTEXT_LEN,
                      ex.input_tokens, CONTEXT_LEN, &acts);
    
    /* Check grad values */
    float* gp = (float*)&grad;
    int nan_count = 0, max_idx = 206912;
    for (int i = 0; i < max_idx; i++) if (isnan(gp[i])) nan_count++;
    printf("NaN gradients: %d/%d\n", nan_count, max_idx);

    if (nan_count < 1000) {
        printf("Grad[0..5]:");
        for (int i = 0; i < 6; i++) printf(" %.6f", gp[i]);
        printf("\n");
    }

    hashmind_apply_gradients(&ctx);

    /* Second step */
    float logits2[VOCAB_SIZE];
    BlockActs acts2;
    hashmind_forward(&model, ex.input_hashes, CONTEXT_LEN,
                     ex.input_tokens, CONTEXT_LEN, logits2, &acts2);
    float loss2 = nn_cross_entropy_loss(logits2, ex.target_token, VOCAB_SIZE);
    printf("Step 2 loss: %.6f\n", loss2);

    /* Check if loss is decreasing */
    printf("Loss delta: %+.6f\n", loss2 - loss);

    textdata_free(&td);
    return 0;
}
