/**
 * train_backprop.c — Phase 3.5: Backpropagation training
 *
 * Full forward pass + CE loss + gradient w.r.t. output.weight
 * SGD weight update, loss tracking over steps.
 *
 * Usage: ./train_backprop [model.gguf] [corpus.bin] [steps]
 *   ENABLE_MOE=1   enable MoE (per-layer lazy load, slow)
 *   MOE_LAYERS=N   limit MoE to first N layers
 *   LR=0.001       learning rate (default: 0.01)
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

// ========== Gradient helpers ==========

// Compute softmax gradient w.r.t. logits, and CE loss
// dlogits = softmax - one_hot(target)
// Returns: CE loss (mean over batch)
static float ce_loss_and_grad(const float *logits, const int *targets,
                               int N, int vocab_size, float *dlogits) {
    float total_loss = 0.0f;
    for (int i = 0; i < N; i++) {
        const float *log_i = logits + i * vocab_size;
        float *dg_i = dlogits + i * vocab_size;
        int t = targets[i];
        
        // Find max for numerical stability
        float max_l = log_i[0];
        for (int j = 1; j < vocab_size; j++)
            if (log_i[j] > max_l) max_l = log_i[j];
        
        // Softmax
        float sum_exp = 0.0f;
        for (int j = 0; j < vocab_size; j++) {
            float e = expf(log_i[j] - max_l);
            dg_i[j] = e;  // temporarily store exp values
            sum_exp += e;
        }
        float inv_sum = 1.0f / (sum_exp + 1e-30f);
        
        // dL/dlogits = softmax - one_hot
        for (int j = 0; j < vocab_size; j++) {
            float soft = dg_i[j] * inv_sum;
            dg_i[j] = soft - (j == t ? 1.0f : 0.0f);
        }
        
        // CE loss = -log(softmax[t])
        float soft_t = expf(log_i[t] - max_l) * inv_sum;
        total_loss += -logf(soft_t + 1e-30f);
    }
    return total_loss / N;
}

// ========== Training step ==========

// Train one step: forward → loss → grad → update output.weight
// Returns: CE loss
static float train_step(wubu_model_t *model,
                        const float *embd, const int *targets,
                        int B, int T, int vocab_size,
                        float *output_weight,  // [D_MODEL, vocab_size] mutable
                        float *hidden_buf,     // [N, D_MODEL] workspace
                        float *logits_buf,     // [N, vocab_size] workspace
                        float *dlogits,        // [N, vocab_size] workspace
                        float *dW,             // [D_MODEL, vocab_size] grad accum
                        float lr) {
    int N = B * T;
    
    // Forward pass: hidden = model(embd)
    memcpy(hidden_buf, embd, N * D_MODEL * sizeof(float));
    wubu_model_forward_from_embd(model, hidden_buf, B, T, hidden_buf);
    
    // Project to vocab: logits = hidden @ output_weight
    #pragma omp parallel for collapse(2) if(N * vocab_size > 100000)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < vocab_size; j++) {
            const float *h = hidden_buf + i * D_MODEL;
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < D_MODEL; k++)
                sum += h[k] * output_weight[j * D_MODEL + k];
            logits_buf[i * vocab_size + j] = sum;
        }
    }
    
    // Loss and gradient dlogits
    float loss = ce_loss_and_grad(logits_buf, targets, N, vocab_size, dlogits);
    
    // Gradient w.r.t. output_weight: dW = sum_i h[i]^T @ dlogits[i]
    // dW[j,k] = sum_i h[i][k] * dlogits[i][j]
    memset(dW, 0, D_MODEL * vocab_size * sizeof(float));
    #pragma omp parallel for if(vocab_size > 1000) collapse(2)
    for (int j = 0; j < vocab_size; j++) {
        for (int k = 0; k < D_MODEL; k++) {
            float sum = 0.0f;
            for (int i = 0; i < N; i++) {
                sum += hidden_buf[i * D_MODEL + k] * dlogits[i * vocab_size + j];
            }
            dW[j * D_MODEL + k] = sum / N;
        }
    }
    
    // SGD update: W = W - lr * dW (with gradient clipping)
    float max_grad_norm = 0.0f;
    for (int64_t idx = 0; idx < (int64_t)D_MODEL * vocab_size; idx++) {
        float g = fabsf(dW[idx]);
        if (g > max_grad_norm) max_grad_norm = g;
    }
    float clip_scale = 1.0f;
    if (max_grad_norm > 1.0f) clip_scale = 1.0f / max_grad_norm;
    for (int64_t idx = 0; idx < (int64_t)D_MODEL * vocab_size; idx++)
        output_weight[idx] -= lr * dW[idx] * clip_scale;
    
    return loss;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1]
        : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *corpus_path = argc > 2 ? argv[2] : "data/train_data.bin";
    int n_steps = argc > 3 ? atoi(argv[3]) : 10;
    
    // Hyperparams
    float lr = 0.01f;
    if (getenv("LR")) lr = atof(getenv("LR"));
    int B = 1, T = 4;
    int N = B * T;
    
    printf("========================================================\n");
    printf("  WuBuText AI — Phase 3.5: Backprop Training\n");
    printf("========================================================\n");
    printf("  Model: %s\n", model_path);
    printf("  Steps: %d, LR: %f, B=%d, T=%d\n", n_steps, lr, B, T);
    fflush(stdout);
    
    // Load tokenizer
    printf("\n--- Loading tokenizer ---\n");
    wubu_tokenizer_t tok;
    double t0 = now_sec();
    if (!wubu_tokenizer_init(&tok, model_path)) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }
    printf("  Vocab: %d tokens in %.2fs\n", tok.vocab_size, now_sec() - t0);
    
    // Load model (without MoE by default)
    printf("\n--- Loading model ---\n");
    wubu_model_t model;
    t0 = now_sec();
    if (!wubu_model_init(&model, model_path)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("  Layers: %d (%d SSM, %d GQA) in %.2fs\n",
           model.n_layers, model.n_layers - model.n_layers/4, model.n_layers/4,
           now_sec() - t0);
    
    // Optional MoE
    if (getenv("ENABLE_MOE")) {
        model.enable_moe = true;
        printf("  MoE: ENABLED\n");
        if (getenv("MOE_LAYERS")) {
            model.moe_max_layers = atoi(getenv("MOE_LAYERS"));
            printf("  MoE: first %d layers\n", model.moe_max_layers);
        }
    } else {
        printf("  MoE: disabled (ENABLE_MOE=1 to enable)\n");
    }
    
    // Load training data
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
    printf("  %d tokens loaded\n", total_tokens);
    
    // Load embeddings file
    const char *embed_path = "data/qwen36_embeddings_c.bin.raw";
    printf("\n--- Loading embeddings ---\n");
    f = fopen(embed_path, "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", embed_path); return 1; }
    fseek(f, 0, SEEK_END);
    long emb_size = ftell(f);
    int emb_tokens = (int)(emb_size / (D_MODEL * sizeof(float)));
    fclose(f);
    printf("  %d embeddings available\n", emb_tokens);
    
    // Load output.weight as mutable float copy (for training)
    printf("\n--- Loading output projection ---\n");
    int vocab_size = tok.vocab_size;
    float *output_weight = (float *)malloc(D_MODEL * vocab_size * sizeof(float));
    {
        gguf_ctx *ctx = gguf_open(model_path);
        if (!ctx) { fprintf(stderr, "Failed to open GGUF\n"); return 1; }
        gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
        if (!t) { fprintf(stderr, "No output.weight\n"); return 1; }
        printf("  output.weight: [%ld, %ld] type=%d\n",
               (long)t->dims[0], (long)t->dims[1], t->ggml_type);
        gguf_read_tensor_f32(ctx, t, output_weight, D_MODEL * vocab_size);
        gguf_close(ctx);
    }
    printf("  output.weight loaded (mutable float copy)\n");
    
    // Allocate buffers
    float *embd = (float *)malloc(N * D_MODEL * sizeof(float));
    float *hidden_buf = (float *)malloc(N * D_MODEL * sizeof(float));
    float *logits_buf = (float *)malloc(N * vocab_size * sizeof(float));
    float *dlogits = (float *)malloc(N * vocab_size * sizeof(float));
    float *dW = (float *)malloc(D_MODEL * vocab_size * sizeof(float));
    
    if (!embd || !hidden_buf || !logits_buf || !dlogits || !dW) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    
    printf("\n========================================================\n");
    printf("  Training: %d steps, B=%d, T=%d\n", n_steps, B, T);
    printf("========================================================\n\n");
    
    double total_time = 0.0;
    for (int step = 0; step < n_steps; step++) {
        // Select batch (cycle through data)
        int start_idx = (step * N) % (total_tokens - N - 1);
        
        // Load embeddings
        f = fopen(embed_path, "rb");
        if (f) {
            for (int i = 0; i < N; i++) {
                int id = tokens[start_idx + i];
                if (id < 0 || id >= emb_tokens) id = 0;
                fseek(f, id * D_MODEL * sizeof(float), SEEK_SET);
                fread(embd + i * D_MODEL, sizeof(float), D_MODEL, f);
            }
            fclose(f);
        } else {
            memset(embd, 0, N * D_MODEL * sizeof(float));
        }
        
        // Targets: predict next token
        int targets[N];
        for (int i = 0; i < N - 1; i++) targets[i] = tokens[start_idx + i + 1];
        targets[N - 1] = 0;
        
        // Training step
        t0 = now_sec();
        float loss = train_step(&model, embd, targets, B, T, vocab_size,
                                output_weight, hidden_buf, logits_buf,
                                dlogits, dW, lr);
        double step_time = now_sec() - t0;
        total_time += step_time;
        
        printf("  Step %3d: loss=%.4f (%.3fs, %.1f tok/s)\n",
               step + 1, loss, step_time, N / step_time);
    }
    
    // Summary
    printf("\n========================================================\n");
    printf("  RESULTS\n");
    printf("========================================================\n");
    printf("  Steps: %d\n", n_steps);
    printf("  Avg time/step: %.3fs (%.1f tok/s)\n",
           total_time / n_steps, N / (total_time / n_steps));
    printf("  Output.weight: trained via SGD (lr=%.4f)\n", lr);
    printf("  Note: output.weight is a float copy; re-quantize for GGUF\n");
    printf("========================================================\n");
    
    // Cleanup
    free(tokens);
    free(embd);
    free(hidden_buf);
    free(logits_buf);
    free(dlogits);
    free(dW);
    free(output_weight);
    wubu_tokenizer_free(&tok);
    wubu_model_free(&model);
    
    return 0;
}
