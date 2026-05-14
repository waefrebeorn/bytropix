#include "wubu_tst.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

// ============================================================
// TST: Token Superposition Training Implementation
// ============================================================

// -----------------------------------------------------------
// In-place softmax (numerically stable)
// -----------------------------------------------------------
void tst_softmax_inplace(float *logits, int N, int V) {
    for (int i = 0; i < N; i++) {
        float *row = logits + i * V;
        // Find max for numerical stability
        float max_val = row[0];
        for (int j = 1; j < V; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < V; j++) {
            row[j] = expf(row[j] - max_val);
            sum += row[j];
        }
        // Normalize
        float inv_sum = 1.0f / (sum + 1e-30f);
        for (int j = 0; j < V; j++) {
            row[j] *= inv_sum;
        }
    }
}

// -----------------------------------------------------------
// Cross-entropy: single row
// -----------------------------------------------------------
float tst_cross_entropy(const float *logits, int V, int label) {
    if (label < 0 || label >= V) {
        fprintf(stderr, "tst_cross_entropy: label %d out of range [0, %d)\n", label, V);
        return 0.0f;
    }
    
    // Allocate temp and softmax
    float *tmp = (float *)malloc(V * sizeof(float));
    if (!tmp) {
        fprintf(stderr, "tst_cross_entropy: allocation failed\n");
        return 0.0f;
    }
    memcpy(tmp, logits, V * sizeof(float));
    
    // Softmax
    float max_val = tmp[0];
    for (int j = 1; j < V; j++) {
        if (tmp[j] > max_val) max_val = tmp[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < V; j++) {
        tmp[j] = expf(tmp[j] - max_val);
        sum += tmp[j];
    }
    float inv_sum = 1.0f / (sum + 1e-30f);
    float prob = tmp[label] * inv_sum;
    
    free(tmp);
    return -logf(prob + 1e-30f);
}

// -----------------------------------------------------------
// Bag embeddings: average s contiguous embeddings
// -----------------------------------------------------------
void tst_bag_embeddings(const float *embeddings, float *bagged,
                        int B, int T, int D, int s) {
    if (T % s != 0) {
        fprintf(stderr, "tst_bag_embeddings: T=%d not divisible by s=%d\n", T, s);
        return;
    }
    
    int T_out = T / s;
    
    for (int b = 0; b < B; b++) {
        for (int ti = 0; ti < T_out; ti++) {
            float *bag_out = bagged + (b * T_out + ti) * D;
            // Zero out
            memset(bag_out, 0, D * sizeof(float));
            // Sum s embeddings
            for (int k = 0; k < s; k++) {
                const float *emb_in = embeddings + (b * T + ti * s + k) * D;
                for (int d = 0; d < D; d++) {
                    bag_out[d] += emb_in[d];
                }
            }
            // Average
            float inv_s = 1.0f / (float)s;
            for (int d = 0; d < D; d++) {
                bag_out[d] *= inv_s;
            }
        }
    }
}

// -----------------------------------------------------------
// Prepare targets: shift left by s-1, split into bags
// -----------------------------------------------------------
int tst_prepare_targets(const int *token_ids, int *targets,
                        int B, int T, int s) {
    // Shift left by s-1: effective length = T - s + 1
    int effective_T = T - s + 1;
    int T_out = effective_T / s;
    
    if (T_out <= 0) return 0;
    
    for (int b = 0; b < B; b++) {
        const int *row_in = token_ids + b * T;
        int *row_out = targets + b * T_out * s;
        
        for (int ti = 0; ti < T_out; ti++) {
            // Bag i consists of: token_ids[s-1 + ti*s] ... token_ids[s-1 + ti*s + s-1]
            for (int k = 0; k < s; k++) {
                int src_idx = (s - 1) + ti * s + k;
                row_out[ti * s + k] = row_in[src_idx];
            }
        }
    }
    
    return T_out;
}

// -----------------------------------------------------------
// MCE loss: mean cross-entropy over s targets per prediction
// -----------------------------------------------------------
bool tst_compute_mce_loss(const float *logits, const int *targets,
                          int B, int T_out, int V, int s,
                          float *loss) {
    if (B <= 0 || T_out <= 0 || V <= 0 || s <= 0) {
        *loss = 0.0f;
        return false;
    }
    
    int N = B * T_out;  // total predictions
    double total_loss = 0.0;
    
    // We compute softmax + CE row by row
    float *softmax_buf = (float *)malloc(V * sizeof(float));
    if (!softmax_buf) {
        *loss = 0.0f;
        return false;
    }
    
    for (int n = 0; n < N; n++) {
        const float *logit_row = logits + n * V;
        const int *tgt_row = targets + n * s;
        
        // Softmax this row
        float max_val = logit_row[0];
        for (int j = 1; j < V; j++) {
            if (logit_row[j] > max_val) max_val = logit_row[j];
        }
        double sum_exp = 0.0;
        for (int j = 0; j < V; j++) {
            softmax_buf[j] = expf(logit_row[j] - max_val);
            sum_exp += softmax_buf[j];
        }
        double inv_sum = 1.0 / (sum_exp + 1e-30);
        
        // CE for each of the s targets
        for (int k = 0; k < s; k++) {
            int label = tgt_row[k];
            if (label < 0 || label >= V) {
                // Clamp out-of-range labels
                if (label < 0) label = 0;
                else label = V - 1;
            }
            double prob = (double)softmax_buf[label] * inv_sum;
            total_loss -= log(prob + 1e-30);
        }
        // Since we double-counted in the loop, each prediction contributes
        // s CE terms.  We'll divide by (N * s) at the end.
    }
    
    free(softmax_buf);
    
    *loss = (float)(total_loss / (double)(N * s));
    return true;
}

// -----------------------------------------------------------
// MCE loss backward: gradient of logits
// -----------------------------------------------------------
void tst_mce_loss_backward(const float *logits, const int *targets,
                           int B, int T_out, int V, int s,
                           float *d_logits) {
    int N = B * T_out;
    
    // Allocate temp for softmax
    float *probs = (float *)malloc(N * V * sizeof(float));
    if (!probs) {
        fprintf(stderr, "tst_mce_backward: alloc failed\n");
        return;
    }
    
    // Compute softmax probabilities
    memcpy(probs, logits, N * V * sizeof(float));
    tst_softmax_inplace(probs, N, V);
    
    // Gradient: d_logits[n, v] = (1/s) * sum_{k=0}^{s-1} (probs[n, v] - delta(v == targets[n,k]))
    float inv_s = 1.0f / (float)s;
    
    for (int n = 0; n < N; n++) {
        float *prob_row = probs + n * V;
        float *d_row = d_logits + n * V;
        const int *tgt_row = targets + n * s;
        
        // Start with copy of probabilities
        memcpy(d_row, prob_row, V * sizeof(float));
        
        // Subtract 1 for each matching target
        for (int k = 0; k < s; k++) {
            int label = tgt_row[k];
            if (label >= 0 && label < V) {
                d_row[label] -= 1.0f;
            }
        }
        
        // Multiply by 1/s
        for (int j = 0; j < V; j++) {
            d_row[j] *= inv_s;
        }
    }
    
    free(probs);
}
