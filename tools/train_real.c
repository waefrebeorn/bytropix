/**
 * train_real.c — Phase 3: Training Pipeline on Real Qwen3.6 Model
 *
 * Loads the 40-layer model + tokenizer + tokenized corpus,
 * runs forward pass through all layers, computes CE loss.
 * Demonstrates the training pipeline is wired end-to-end.
 */

#include "wubu_model.h"
#include "wubu_tokenizer.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Cross-entropy loss (same as train_stub.c)
static float cross_entropy_loss(const float *logits, const int *targets,
                                 int N, int vocab_size) {
    float total_loss = 0.0f;
    for (int i = 0; i < N; i++) {
        float max_l = logits[i * vocab_size];
        for (int j = 1; j < vocab_size; j++)
            if (logits[i * vocab_size + j] > max_l)
                max_l = logits[i * vocab_size + j];
        float sum_exp = 0.0f;
        for (int j = 0; j < vocab_size; j++)
            sum_exp += expf(logits[i * vocab_size + j] - max_l);
        int t = targets[i];
        float softmax_t = expf(logits[i * vocab_size + t] - max_l) / (sum_exp + 1e-30f);
        total_loss += -logf(softmax_t + 1e-30f);
    }
    return total_loss / N;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1]
        : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *corpus_path = argc > 2 ? argv[2] : "data/train_data.bin";
    const char *embed_path = "data/qwen36_embeddings_c.bin";

    printf("========================================================\n");
    printf("  WuBuText AI — Phase 3 Real Model Training Pipeline\n");
    printf("========================================================\n");
    printf("  Model: %s\n", model_path);
    printf("  Corpus: %s\n", corpus_path);
    fflush(stdout);

    // ===== Load tokenizer =====
    printf("\n--- Loading tokenizer ---\n");
    wubu_tokenizer_t tok;
    double t0 = now_sec();
    if (!wubu_tokenizer_init(&tok, model_path)) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }
    double t_tok = now_sec() - t0;
    printf("  Vocab: %d tokens (%d merges) in %.2fs\n",
           tok.vocab_size, tok.n_merges, t_tok);

    // ===== Load model =====
    printf("\n--- Loading model ---\n");
    wubu_model_t model;
    t0 = now_sec();
    if (!wubu_model_init(&model, model_path)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    double t_model = now_sec() - t0;
    printf("  Layers: %d (%d SSM, %d GQA) in %.2fs\n",
           model.n_layers,
           model.n_layers - model.n_layers / 4,
           model.n_layers / 4,
           t_model);

    // ===== Load tokenized training data =====
    printf("\n--- Loading training data ---\n");
    FILE *f = fopen(corpus_path, "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", corpus_path); return 1; }
    fseek(f, 0, SEEK_END);
    long data_size = ftell(f);
    int total_tokens = (int)(data_size / sizeof(int));
    fseek(f, 0, SEEK_SET);
    int *tokens = (int *)malloc(data_size);
    fread(tokens, 1, data_size, f);
    fclose(f);
    printf("  %d tokens loaded (%ld bytes)\n", total_tokens, data_size);

    // ===== Load embeddings =====
    printf("\n--- Loading embeddings ---\n");
    f = fopen(embed_path, "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", embed_path); return 1; }
    fseek(f, 0, SEEK_END);
    long emb_size = ftell(f);
    int emb_tokens = (int)(emb_size / (D_MODEL * sizeof(float)));
    fclose(f);
    printf("  %d embeddings available (%ld MB)\n", emb_tokens, emb_size / (1024*1024));

    // ===== Load output weight (for logit projection) =====
    printf("\n--- Loading output projection ---\n");
    gguf_ctx *ctx = gguf_open(model_path);
    float *output_weight = NULL;
    if (ctx) {
        gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
        if (t) {
            printf("  output.weight: [%ld, %ld] type=%d\n",
                   (long)t->dims[0], (long)t->dims[1], t->ggml_type);
            output_weight = (float *)malloc(tok.vocab_size * D_MODEL * sizeof(float));
            t0 = now_sec();
            int nread = gguf_read_tensor_f32(ctx, t, output_weight, tok.vocab_size * D_MODEL);
            double t_ow = now_sec() - t0;
            printf("  output.weight loaded (%d elems) in %.2fs\n", nread, t_ow);
        }
        gguf_close(ctx);
    }

    // ===== Training configuration =====
    int B = 1, T = 4;  // small batch for test
    int N = B * T;
    int total_batches = total_tokens / N;
    if (total_batches < 1) { fprintf(stderr, "Not enough tokens for one batch\n"); return 1; }

    printf("\n--- Training config ---\n");
    printf("  B=%d, T=%d, N=%d tokens/batch\n", B, T, N);
    printf("  Total batches available: %d\n", total_batches);
    printf("  Vocab: %d\n", tok.vocab_size);

    // ===== Run forward pass =====
    printf("\n--- Running forward pass (layer loop) ---\n");

    // Allocate buffers
    float *embd = (float *)malloc(N * D_MODEL * sizeof(float));
    float *logits = (float *)malloc(N * tok.vocab_size * sizeof(float));
    int *targets = (int *)malloc(N * sizeof(int));

    // Use first N tokens from corpus
    // Look up embeddings
    if (model.use_embedding_file) {
        f = fopen(embed_path, "rb");
        if (f) {
            for (int i = 0; i < N; i++) {
                int id = tokens[i];
                if (id < 0 || id >= emb_tokens) id = 0;
                fseek(f, id * D_MODEL * sizeof(float), SEEK_SET);
                fread(embd + i * D_MODEL, sizeof(float), D_MODEL, f);
            }
            fclose(f);
        } else {
            memset(embd, 0, N * D_MODEL * sizeof(float));
        }
    }

    // Targets: predict next token
    for (int i = 0; i < N - 1; i++) targets[i] = tokens[i + 1];
    targets[N - 1] = 0;

    // Warm-up run
    printf("  Warm-up: running forward pass...\n");
    wubu_model_forward_from_embd(&model, embd, B, T, logits);
    printf("  Warm-up done.\n");

    // Timed runs
    int n_runs = 3;
    double total_forward = 0.0, total_loss = 0.0;
    for (int run = 0; run < n_runs; run++) {
        t0 = now_sec();
        wubu_model_forward_from_embd(&model, embd, B, T, logits);
        double t_fwd = now_sec() - t0;

        double t_l = now_sec();
        float loss = 0.0f;

        // Project to vocab space if output_weight loaded
        // Project to vocab space and compute CE loss
        if (output_weight) {
            float total_loss_val = 0.0f;
            for (int i = 0; i < N; i++) {
                const float *h = logits + i * D_MODEL;
                int target = targets[i];
                
                // Compute all 248K vocab logits in streaming fashion
                // hidden[D] @ output_weight[D, V] = logits[V]
                double max_l = -1e30;
                double target_logit = 0.0;
                
                for (int j = 0; j < tok.vocab_size; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < D_MODEL; k++)
                        sum += (double)h[k] * (double)output_weight[j * D_MODEL + k];
                    if (j == target) target_logit = sum;
                    if (sum > max_l) max_l = sum;
                }
                
                // log-sum-exp = max_l + log(sum(exp(logits - max_l)))
                double sum_exp = 0.0;
                for (int j = 0; j < tok.vocab_size; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < D_MODEL; k++)
                        sum += (double)h[k] * (double)output_weight[j * D_MODEL + k];
                    sum_exp += exp(sum - max_l);
                }
                double log_sum_exp = max_l + log(sum_exp);
                
                // CE = -(target_logit - log_sum_exp)
                double ce = -(target_logit - log_sum_exp);
                total_loss_val += (float)ce;
            }
            loss = total_loss_val / N;
            printf("  CE loss: %.4f\n", loss);
        }
        
        // Log first few logit values
        if (run == 0) {
            printf("  Logits[0:8] for token 0:");
            for (int i = 0; i < 8 && i < tok.vocab_size; i++)
                printf(" %+.4f", logits[i]);
            printf("\n");
            printf("  Logits range: [%.4f, %.4f]\n",
                   logits[0], logits[tok.vocab_size - 1]);
        }

        total_forward += t_fwd;
        printf("  Run %d: forward = %.3fs (%.1f tok/s)\n",
               run + 1, t_fwd, N / t_fwd);
    }

    // ===== Summary =====
    printf("\n========================================================\n");
    printf("  RESULTS\n");
    printf("========================================================\n");
    printf("  Avg forward time: %.3fs\n", total_forward / n_runs);
    printf("  Throughput: %.1f tok/s (B=%d, T=%d)\n",
           N / (total_forward / n_runs), B, T);
    printf("  Model: 40 layers (30 SSM + 10 GQA)\n");
    printf("  Next: add backprop + MoE forward\n");
    printf("========================================================\n");

    // Cleanup
    free(tokens);
    free(embd);
    free(logits);
    free(targets);
    free(output_weight);
    wubu_tokenizer_free(&tok);
    wubu_model_free(&model);

    return 0;
}
