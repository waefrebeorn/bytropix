/**
 * train.c — HashMind Training Entry Point (pure C)
 *
 * Usage: ./train [options]
 *   --data  <file>   Training data text file (default: training_sample.txt)
 *   --lr   <float>   Learning rate (default: 0.001)
 *   --epochs <int>   Number of epochs (default: 100)
 *   --save <path>    Save checkpoint path (default: hashmind_model.bin)
 *   --generate <int> Generate sample every N steps (default: 500)
 *   --test <string>  Test string for eval (default: "Hello World")
 */
#include "tokenizer.h"
#include "hashmind_model.h"
#include "hashmind_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    const char* data_path = "training_sample.txt";
    const char* save_path = "hashmind_model.bin";
    const char* test_str = "Hello World";
    float lr = 0.001f;
    float momentum = 0.9f;
    float weight_decay = 1e-4f;
    int epochs = 100;
    int gen_every = 500;
    int gen_max = 64;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i+1 < argc) data_path = argv[++i];
        else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) lr = atof(argv[++i]);
        else if (strcmp(argv[i], "--epochs") == 0 && i+1 < argc) epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--save") == 0 && i+1 < argc) save_path = argv[++i];
        else if (strcmp(argv[i], "--gen-every") == 0 && i+1 < argc) gen_every = atoi(argv[++i]);
        else if (strcmp(argv[i], "--generate") == 0 && i+1 < argc) gen_max = atoi(argv[++i]);
        else if (strcmp(argv[i], "--test") == 0 && i+1 < argc) test_str = argv[++i];
    }

    /* Init tokenizer */
    Tokenizer tok;
    tokenizer_init(&tok);
    printf("Vocab: %d tokens\n", tok.vocab_size);

    /* Load training data */
    FILE* fdata = fopen(data_path, "rb");
    if (!fdata) { fprintf(stderr, "Cannot open %s\n", data_path); return 1; }
    fseek(fdata, 0, SEEK_END);
    long fsize = ftell(fdata);
    rewind(fdata);
    char* text = (char*)malloc(fsize + 1);
    fread(text, 1, fsize, fdata);
    text[fsize] = '\0';
    fclose(fdata);
    printf("Data: %ld bytes from %s\n", fsize, data_path);

    /* Init model */
    HashMindModel model;
    HashMindGrad grad;
    long nparams = hashmind_param_count();
    /* Momentum buffer needs 2x for moment1 + moment2 */
    float* vel_buf = (float*)calloc(nparams * 2, sizeof(float));
    HashMindMomentum* vel = (HashMindMomentum*)vel_buf;
    hashmind_model_init(&model);
    hashmind_zero_grad(&grad);

    printf("Model: %ld parameters (%zu bytes, momentum %zu bytes)\n",
           nparams, sizeof(HashMindModel), nparams * 2 * sizeof(float));

    TrainCtx ctx;
    ctx.lr = lr;
    ctx.weight_decay = weight_decay;
    ctx.step = 0;
    ctx.model = &model;
    ctx.grad = &grad;
    ctx.vel = vel;

    /* Init data */
    TextData td;
    textdata_init(&td, &tok, text);

    /* Training loop */
    double t0 = now_sec();
    double t_last = t0;
    printf("\n=== Training ===\n");
    printf("Epochs: %d, LR: %.4f, Momentum: %.2f, WD: %.6f\n", epochs, lr, momentum, weight_decay);
    printf("Steps/epoch: ~%d\n", td.num_tokens - CONTEXT_LEN - 1);
    fflush(stdout);

    int total_steps = 0;
    float running_loss = 0;
    int loss_count = 0;

    for (int ep = 0; ep < epochs; ep++) {
        textdata_reset(&td);
        TrainExample ex;
        int steps_in_epoch = 0;

        while (textdata_next(&td, &ex)) {
            /* Forward */
            float logits[VOCAB];
            BlockActs acts;
            hashmind_forward(&model, ex.input_hashes, CONTEXT_LEN,
                             ex.input_tokens, CONTEXT_LEN, logits, &acts);

            /* Loss */
            float loss = nn_cross_entropy_loss(logits, ex.target_token, VOCAB);
            running_loss += loss;
            loss_count++;
            total_steps++;

            /* Backward */
            float dlogits[VOCAB];
            nn_cross_entropy_grad(logits, ex.target_token, VOCAB, dlogits);
            hashmind_backward(&model, &grad, dlogits, ex.input_hashes, CONTEXT_LEN,
                              ex.input_tokens, CONTEXT_LEN, &acts);

            /* Update */
            hashmind_apply_gradients(&ctx);

            steps_in_epoch++;

            /* Logging */
            if (total_steps % 100 == 0) {
                double t_now = now_sec();
                double dt = t_now - t_last;
                double steps_per_sec = 100.0 / dt;
                printf("[ep %d/%d step %d] loss=%.4f (%.1f steps/s, %ld params, %s)\n",
                       ep+1, epochs, total_steps, running_loss / loss_count,
                       steps_per_sec, nparams, dt < 5 ? "cpu" : "thrashing");
                running_loss = 0;
                loss_count = 0;
                t_last = t_now;
                fflush(stdout);
            }

            /* Generate sample */
            if (total_steps % gen_every == 0) {
                char gen_buf[1024];
                textdata_generate_sample(&model, &tok, test_str, gen_max, 0.8f, gen_buf, 1024);
                printf("  Gen[%d]: %s\n", total_steps, gen_buf);
                fflush(stdout);
            }
        }

        /* End of epoch */
        double t_now = now_sec();
        double epoch_time = t_now - t_last;
        printf("--- Epoch %d done: %d steps, %.2fs, avg %.1f steps/s ---\n",
               ep+1, steps_in_epoch, epoch_time, steps_in_epoch / epoch_time);
        fflush(stdout);
    }

    double total_time = now_sec() - t0;
    printf("\n=== Training Complete ===\n");
    printf("Total steps: %d, Time: %.2fs (%.1f steps/s avg)\n",
           total_steps, total_time, total_steps / total_time);
    fflush(stdout);

    /* Save */
    if (hashmind_save(&model, save_path) == 0) {
        printf("Model saved to %s (%zu bytes)\n", save_path, sizeof(HashMindModel));
    } else {
        fprintf(stderr, "Failed to save model to %s\n", save_path);
    }

    /* Final generation */
    printf("\n=== Final Generation ===\n");
    char gen_buf[4096];
    textdata_generate_sample(&model, &tok, test_str, 256, 0.7f, gen_buf, 4096);
    printf("%s\n", gen_buf);
    fflush(stdout);

    /* Cleanup */
    textdata_free(&td);
    free(text);
    free(vel_buf);

    return 0;
}
