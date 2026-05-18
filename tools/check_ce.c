#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    
    // Load output.weight  
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    int64_t total = t->dims[0] * t->dims[1];
    float *w = (float*)malloc(total * sizeof(float));
    gguf_read_tensor_f32(ctx, t, w, total);
    
    // Create a simulated hidden state: all -3.0 (matching the debug output)
    float h[2048];
    for (int i = 0; i < 2048; i++) h[i] = -3.0f;
    
    // Compute logits for first 10000 vocab entries
    printf("Simulating logits with h = [-3, -3, ..., -3]:\n");
    double max_l = -1e30, min_l = 1e30, sum_l = 0;
    int n_check = 50000;
    if (n_check > t->dims[1]) n_check = t->dims[1];
    
    for (int j = 0; j < n_check; j++) {
        double sum = 0;
        for (int k = 0; k < 2048; k++)
            sum += h[k] * w[j * 2048 + k];
        if (sum > max_l) max_l = sum;
        if (sum < min_l) min_l = sum;
        sum_l += sum;
    }
    printf("  First %d logits: mean=%.2f min=%.2f max=%.2f\n", n_check, sum_l/n_check, min_l, max_l);
    
    // Show logit distribution for first 100
    printf("  First 20 logits:");
    for (int j = 0; j < 20; j++) {
        double sum = 0;
        for (int k = 0; k < 2048; k++)
            sum += h[k] * w[j * 2048 + k];
        printf(" %.1f", sum);
    }
    printf("\n");
    
    // Check logits for specific tokens
    int targets[] = {20, 100, 1000, 10000, 50000, 100000, 150000, 200000};
    for (int ti = 0; ti < 8; ti++) {
        int j = targets[ti];
        if (j >= t->dims[1]) continue;
        double sum = 0;
        for (int k = 0; k < 2048; k++)
            sum += h[k] * w[j * 2048 + k];
        printf("  Logit[%d] = %.2f\n", j, sum);
    }
    
    // Full CE computation for one token with this hidden state
    // Using stable softmax
    int V = t->dims[1];
    int target_id = 20;  // arbitrary token ID from corpus
    
    // Find max
    max_l = -1e30;
    for (int j = 0; j < V; j++) {
        double sum = 0;
        for (int k = 0; k < 2048; k++)
            sum += h[k] * w[j * 2048 + k];
        if (sum > max_l) max_l = sum;
    }
    printf("\n  Max logit over %d vocab: %.2f\n", V, max_l);
    
    // Compute sum_exp with stability
    double sum_exp = 0;
    for (int j = 0; j < V; j++) {
        double sum = 0;
        for (int k = 0; k < 2048; k++)
            sum += h[k] * w[j * 2048 + k];
        double e = exp(sum - max_l);
        if (isinf(e) || isnan(e)) {
            printf("  Overflow at j=%d: sum=%.2f max_l=%.2f diff=%.2f\n", j, sum, max_l, sum-max_l);
            break;
        }
        sum_exp += e;
    }
    double log_sum_exp = max_l + log(sum_exp);
    
    // Target logit
    double target_val = 0;
    for (int k = 0; k < 2048; k++)
        target_val += h[k] * w[target_id * 2048 + k];
    
    double ce = -(target_val - log_sum_exp);
    printf("  Target[%d]: %.2f\n", target_id, target_val);
    printf("  Log-sum-exp: %.2f\n", log_sum_exp);
    printf("  CE loss: %.2f\n", ce);
    printf("  Expected (random): %.2f\n", log((double)V));
    
    free(w);
    gguf_close(ctx);
    return 0;
}
