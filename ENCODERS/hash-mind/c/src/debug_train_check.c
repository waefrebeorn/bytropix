#include "tokenizer.h"
#include "hashmind_model.h"
#include "hashmind_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

    /* Load training_sample.txt */
    FILE* f = fopen("training_sample.txt", "rb");
    if (!f) { printf("Cannot open\n"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);
    char* text = (char*)malloc(fsize + 1);
    fread(text, 1, fsize, f);
    text[fsize] = '\0';
    fclose(f);
    printf("Data: %ld bytes\n", fsize);

    TextData td;
    textdata_init(&td, &tok, text);
    printf("Tokens: %d, Hashes: computed from tokens\n", td.num_tokens);

    /* Run 10 steps and check */
    for (int step = 0; step < 10; step++) {
        TrainExample ex;
        if (!textdata_next(&td, &ex)) { printf("No more data\n"); break; }

        float logits[VOCAB_SIZE];
        BlockActs acts;
        hashmind_forward(&model, ex.input_hashes, CONTEXT_LEN,
                         ex.input_tokens, CONTEXT_LEN, logits, &acts);

        float loss = nn_cross_entropy_loss(logits, ex.target_token, VOCAB_SIZE);
        
        /* Check for nan */
        if (isnan(loss) || isinf(loss)) {
            printf("Step %d: LOSS = NaN! Target=%d\n", step, ex.target_token);
            printf("  Input tokens[0..5]: %d %d %d %d %d %d\n",
                   ex.input_tokens[0], ex.input_tokens[1], ex.input_tokens[2],
                   ex.input_tokens[3], ex.input_tokens[4], ex.input_tokens[5]);
            printf("  Hashes[0..3]: %u %u %u %u\n",
                   ex.input_hashes[0], ex.input_hashes[1], ex.input_hashes[2], ex.input_hashes[3]);
            printf("  logits[0..5]: %.4f %.4f %.4f %.4f %.4f %.4f\n",
                   logits[0], logits[1], logits[2], logits[3], logits[4], logits[5]);
            return 1;
        }

        float dlogits[VOCAB_SIZE];
        nn_cross_entropy_grad(logits, ex.target_token, VOCAB_SIZE, dlogits);
        hashmind_backward(&model, &grad, dlogits, ex.input_hashes, CONTEXT_LEN,
                          ex.input_tokens, CONTEXT_LEN, &acts);
        hashmind_apply_gradients(&ctx);

        printf("Step %d: loss=%.4f\n", step, loss);
    }

    printf("\nAll OK - no NaN in first 10 steps\n");
    
    /* Run 100 more steps silently */
    int nan_steps = 0;
    for (int step = 10; step < 1000; step++) {
        TrainExample ex;
        if (!textdata_next(&td, &ex)) { textdata_reset(&td); textdata_next(&td, &ex); }

        float logits[VOCAB_SIZE];
        BlockActs acts;
        hashmind_forward(&model, ex.input_hashes, CONTEXT_LEN,
                         ex.input_tokens, CONTEXT_LEN, logits, &acts);
        float loss = nn_cross_entropy_loss(logits, ex.target_token, VOCAB_SIZE);
        if (isnan(loss) || isinf(loss)) { nan_steps++; continue; }

        float dlogits[VOCAB_SIZE];
        nn_cross_entropy_grad(logits, ex.target_token, VOCAB_SIZE, dlogits);
        hashmind_backward(&model, &grad, dlogits, ex.input_hashes, CONTEXT_LEN,
                          ex.input_tokens, CONTEXT_LEN, &acts);
        hashmind_apply_gradients(&ctx);

        if (step % 100 == 0) printf("Step %d: loss=%.4f\n", step, loss);
    }
    printf("NaN steps: %d\n", nan_steps);

    free(text);
    textdata_free(&td);
    return 0;
}
