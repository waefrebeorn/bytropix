#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

// ============================================================
// Hyperparameters
// ============================================================
#define BATCH       1
#define TOKENS      4
#define D_MODEL     2048
#define D_FFN       512
#define N_EXPERTS   256
#define TOP_K       8

// ============================================================
// Helper: matrix multiply C[M,N] = A[M,K] * B[K,N] (row-major)
// ============================================================
static void matmul(const float *A, const float *B, int M, int K, int N, float *C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += (double)A[i * K + k] * (double)B[k * N + j];
            }
            C[i * N + j] = (float)sum;
        }
    }
}

// ============================================================
// Helper: SiLU activation (in-place)
// ============================================================
static void silu(int n, float *x) {
    for (int i = 0; i < n; i++) {
        float s = 1.0f / (1.0f + expf(-x[i])); // sigmoid
        x[i] = x[i] * s;
    }
}

// ============================================================
// Helper: softmax over last dimension (in-place)
// ============================================================
static void softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        // Find max for numerical stability
        float maxv = row[0];
        for (int c = 1; c < cols; c++) if (row[c] > maxv) maxv = row[c];
        double sum = 0.0;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - maxv);
            sum += row[c];
        }
        double inv_sum = 1.0 / sum;
        for (int c = 0; c < cols; c++) row[c] = (float)(row[c] * inv_sum);
    }
}

// ============================================================
// Helper: top-k selection (returns indices and weights)
// ============================================================
typedef struct { float weight; int index; } topk_entry;

static int cmp_topk(const void *a, const void *b) {
    float wa = ((const topk_entry*)a)->weight;
    float wb = ((const topk_entry*)b)->weight;
    if (wa > wb) return -1;
    if (wa < wb) return 1;
    return 0;
}

// ============================================================
// Main: MoE forward pass test
// ============================================================
int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int layer_idx = argc > 2 ? atoi(argv[2]) : 0;
    
    printf("=== MoE Forward Pass Test ===\n");
    printf("Model: %s\n", model_path);
    printf("Layer: %d\n", layer_idx);
    printf("B=%d T=%d D=%d N_EXPERTS=%d TOP_K=%d\n\n", BATCH, TOKENS, D_MODEL, N_EXPERTS, TOP_K);
    
    // --------------------------------------------------
    // 1) Open GGUF file
    // --------------------------------------------------
    gguf_ctx *ctx = gguf_open(model_path);
    if (!ctx) {
        fprintf(stderr, "ERROR: Cannot open %s\n", model_path);
        return 1;
    }
    
    // --------------------------------------------------
    // 2) Load all 8 MoE tensors for layer 0
    // --------------------------------------------------
    char name[256];
    gguf_tensor_info *t;
    
    // Tensor 1: ffn_gate_inp.weight [2048, 256] F32 — router weights
    float *ffn_gate_inp = NULL;
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 1; }
    ffn_gate_inp = (float *)malloc(D_MODEL * N_EXPERTS * sizeof(float));
    int ok = gguf_read_tensor_f32(ctx, t, ffn_gate_inp, D_MODEL * N_EXPERTS);
    printf("  Loaded %s [%ld,%ld] type=%d ok=%d\n", name, (long)t->dims[0], (long)t->dims[1], t->ggml_type, ok);
    
    // Tensor 2: ffn_gate_inp_shexp.weight [2048] F32 — shared expert gate
    float *ffn_gate_inp_shexp = NULL;
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp_shexp.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 1; }
    ffn_gate_inp_shexp = (float *)malloc(D_MODEL * sizeof(float));
    ok = gguf_read_tensor_f32(ctx, t, ffn_gate_inp_shexp, D_MODEL);
    printf("  Loaded %s [%ld] type=%d ok=%d\n", name, (long)t->dims[0], t->ggml_type, ok);
    
    // Tensor 3: ffn_gate_exps.weight [2048, 512, 256] IQ2_XXS
    float *ffn_gate_exps = NULL;
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 1; }
    int64_t gate_exps_elems = D_MODEL * D_FFN * N_EXPERTS;
    ffn_gate_exps = (float *)malloc(gate_exps_elems * sizeof(float));
    ok = gguf_read_tensor_f32(ctx, t, ffn_gate_exps, gate_exps_elems);
    printf("  Loaded %s [%ld,%ld,%ld] type=%d ok=%d\n", name, (long)t->dims[0], (long)t->dims[1], (long)t->dims[2], t->ggml_type, ok);
    
    // Tensor 4: ffn_up_exps.weight [2048, 512, 256] IQ2_XXS
    float *ffn_up_exps = NULL;
    snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 1; }
    ffn_up_exps = (float *)malloc(gate_exps_elems * sizeof(float));
    ok = gguf_read_tensor_f32(ctx, t, ffn_up_exps, gate_exps_elems);
    printf("  Loaded %s [%ld,%ld,%ld] type=%d ok=%d\n", name, (long)t->dims[0], (long)t->dims[1], (long)t->dims[2], t->ggml_type, ok);
    
    // Tensor 5: ffn_down_exps.weight [512, 2048, 256] IQ2_S
    float *ffn_down_exps = NULL;
    snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 1; }
    int64_t down_exps_elems = D_FFN * D_MODEL * N_EXPERTS;
    ffn_down_exps = (float *)malloc(down_exps_elems * sizeof(float));
    ok = gguf_read_tensor_f32(ctx, t, ffn_down_exps, down_exps_elems);
    printf("  Loaded %s [%ld,%ld,%ld] type=%d ok=%d\n", name, (long)t->dims[0], (long)t->dims[1], (long)t->dims[2], t->ggml_type, ok);
    
    // Tensor 6: ffn_gate_shexp.weight [2048, 512] Q5_K
    float *ffn_gate_shexp = NULL;
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 1; }
    ffn_gate_shexp = (float *)malloc(D_MODEL * D_FFN * sizeof(float));
    ok = gguf_read_tensor_f32(ctx, t, ffn_gate_shexp, D_MODEL * D_FFN);
    printf("  Loaded %s [%ld,%ld] type=%d ok=%d\n", name, (long)t->dims[0], (long)t->dims[1], t->ggml_type, ok);
    
    // Tensor 7: ffn_up_shexp.weight [2048, 512] Q5_K
    float *ffn_up_shexp = NULL;
    snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 1; }
    ffn_up_shexp = (float *)malloc(D_MODEL * D_FFN * sizeof(float));
    ok = gguf_read_tensor_f32(ctx, t, ffn_up_shexp, D_MODEL * D_FFN);
    printf("  Loaded %s [%ld,%ld] type=%d ok=%d\n", name, (long)t->dims[0], (long)t->dims[1], t->ggml_type, ok);
    
    // Tensor 8: ffn_down_shexp.weight [512, 2048] Q6_K
    float *ffn_down_shexp = NULL;
    snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (!t) { fprintf(stderr, "Cannot find %s\n", name); return 1; }
    ffn_down_shexp = (float *)malloc(D_FFN * D_MODEL * sizeof(float));
    ok = gguf_read_tensor_f32(ctx, t, ffn_down_shexp, D_FFN * D_MODEL);
    printf("  Loaded %s [%ld,%ld] type=%d ok=%d\n", name, (long)t->dims[0], (long)t->dims[1], t->ggml_type, ok);
    
    // --------------------------------------------------
    // 3) Create random input [BATCH, TOKENS, D_MODEL]
    // --------------------------------------------------
    srand(42);
    float *x = (float *)malloc(BATCH * TOKENS * D_MODEL * sizeof(float));
    for (int i = 0; i < BATCH * TOKENS * D_MODEL; i++) {
        x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
    }
    printf("\n  Created random input x[0:8]:");
    for (int i = 0; i < 8 && i < BATCH*TOKENS*D_MODEL; i++) printf(" %+.6f", x[i]);
    printf("\n");
    
    // --------------------------------------------------
    // 4) MoE Forward Pass
    // --------------------------------------------------
    int BT = BATCH * TOKENS;
    float *output = (float *)calloc(BT * D_MODEL, sizeof(float));
    
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    
    // --- Step A: Router logits ---
    // router_logits = x @ ffn_gate_inp^T  => [BT, N_EXPERTS]
    float *router_logits = (float *)malloc(BT * N_EXPERTS * sizeof(float));
    // ffn_gate_inp is [D_MODEL, N_EXPERTS] = [2048, 256]
    // x[BT, D_MODEL] @ ffn_gate_inp[D_MODEL, N_EXPERTS]
    matmul(x, ffn_gate_inp, BT, D_MODEL, N_EXPERTS, router_logits);
    
    // --- Step B: Softmax ---
    softmax(router_logits, BT, N_EXPERTS);
    
    // --- Step C: Top-K selection per token ---
    // For each token (BT total), find top-k experts
    topk_entry *entries = (topk_entry *)malloc(N_EXPERTS * sizeof(topk_entry));
    int *topk_indices = (int *)malloc(BT * TOP_K * sizeof(int));
    float *topk_weights = (float *)malloc(BT * TOP_K * sizeof(float));
    
    for (int b = 0; b < BT; b++) {
        float *row = router_logits + b * N_EXPERTS;
        for (int e = 0; e < N_EXPERTS; e++) {
            entries[e].weight = row[e];
            entries[e].index = e;
        }
        qsort(entries, N_EXPERTS, sizeof(topk_entry), cmp_topk);
        // Normalize top-k weights
        float sum = 0.0f;
        for (int k = 0; k < TOP_K; k++) sum += entries[k].weight;
        float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int k = 0; k < TOP_K; k++) {
            topk_indices[b * TOP_K + k] = entries[k].index;
            topk_weights[b * TOP_K + k] = entries[k].weight * inv_sum;
        }
    }
    
    printf("\n  Top-8 router probs for token 0:");
    for (int k = 0; k < TOP_K; k++)
        printf(" e%d:%.4f", topk_indices[k], topk_weights[k]);
    printf("\n");
    
    // --- Step D: Expert compute ---
    // For each token, for each of its top-k experts:
    //   gate = SiLU(x @ gate_exps[:,:,e]^T)    [BT, D_FFN]
    //   up   = x @ up_exps[:,:,e]^T            [BT, D_FFN]
    //   hidden = gate * up                      [BT, D_FFN]
    //   out = hidden @ down_exps[:,:,e]^T       [BT, D_MODEL]
    //   output += weight * out
    
    float *buf_gate = (float *)malloc(D_FFN * sizeof(float));
    float *buf_up = (float *)malloc(D_FFN * sizeof(float));
    float *buf_hidden = (float *)malloc(D_FFN * sizeof(float));
    float *buf_expert_out = (float *)malloc(D_MODEL * sizeof(float));
    
    for (int bt = 0; bt < BT; bt++) {
        float *x_tok = x + bt * D_MODEL;
        
        for (int k = 0; k < TOP_K; k++) {
            int e = topk_indices[bt * TOP_K + k];
            float w = topk_weights[bt * TOP_K + k];
            
            // Expert gate: x_tok[1,D_MODEL] @ gate_exps[:,:,e][D_MODEL, D_FFN] -> gate[1, D_FFN]
            float *gate_w = ffn_gate_exps + e * D_MODEL * D_FFN; // expert e's gate weight
            matmul(x_tok, gate_w, 1, D_MODEL, D_FFN, buf_gate);
            
            // Expert up: x_tok @ up_exps[:,:,e]
            float *up_w = ffn_up_exps + e * D_MODEL * D_FFN;
            matmul(x_tok, up_w, 1, D_MODEL, D_FFN, buf_up);
            
            // SiLU(gate) * up
            memcpy(buf_hidden, buf_gate, D_FFN * sizeof(float));
            silu(D_FFN, buf_hidden);
            for (int i = 0; i < D_FFN; i++) buf_hidden[i] *= buf_up[i];
            
            // hidden @ down_exps[:,:,e]^T  -- down_exps is [D_FFN, D_MODEL, N_EXPERTS]
            float *down_w = ffn_down_exps + e * D_FFN * D_MODEL;
            matmul(buf_hidden, down_w, 1, D_FFN, D_MODEL, buf_expert_out);
            
            // Weighted sum into output
            for (int i = 0; i < D_MODEL; i++) {
                output[bt * D_MODEL + i] += w * buf_expert_out[i];
            }
        }
    }
    
    // --- Step E: Shared expert ---
    // shared_gate = SiLU(x @ gate_shexp^T) * (x @ up_shexp^T)
    // shared_down = shared_gate @ down_shexp^T
    // output += shared_down
    
    float *shared_gate = (float *)malloc(D_FFN * sizeof(float));
    float *shared_up = (float *)malloc(D_FFN * sizeof(float));
    float *shared_out = (float *)malloc(D_MODEL * sizeof(float));
    
    for (int bt = 0; bt < BT; bt++) {
        float *x_tok = x + bt * D_MODEL;
        
        // shared_gate = x @ ffn_gate_shexp^T [D_MODEL, D_FFN]
        matmul(x_tok, ffn_gate_shexp, 1, D_MODEL, D_FFN, shared_gate);
        
        // shared_up = x @ ffn_up_shexp^T [D_MODEL, D_FFN]
        matmul(x_tok, ffn_up_shexp, 1, D_MODEL, D_FFN, shared_up);
        
        // SiLU(gate) * up
        silu(D_FFN, shared_gate);
        for (int i = 0; i < D_FFN; i++) shared_gate[i] *= shared_up[i];
        
        // shared_gate @ ffn_down_shexp^T [D_FFN, D_MODEL]
        matmul(shared_gate, ffn_down_shexp, 1, D_FFN, D_MODEL, shared_out);
        
        // Add to output
        for (int i = 0; i < D_MODEL; i++) {
            output[bt * D_MODEL + i] += shared_out[i];
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    
    // --------------------------------------------------
    // 5) Print results
    // --------------------------------------------------
    printf("\n=== Results ===\n");
    printf("Output[0:8] for token 0:");
    for (int i = 0; i < 8; i++) printf(" %+.6f", output[i]);
    printf("\n");
    printf("Output[0:8] for token 1:");
    for (int i = 0; i < 8; i++) printf(" %+.6f", output[D_MODEL + i]);
    printf("\n");
    
    float min_v = 1e30f, max_v = -1e30f, mean = 0.0f;
    for (int i = 0; i < BT * D_MODEL; i++) {
        if (output[i] < min_v) min_v = output[i];
        if (output[i] > max_v) max_v = output[i];
        mean += output[i];
    }
    mean /= (BT * D_MODEL);
    printf("Output range: [%e, %e], mean: %e\n", min_v, max_v, mean);
    
    printf("\n=== Benchmark ===\n");
    printf("Time: %.3f ms\n", elapsed * 1000.0);
    printf("Tokens: %d\n", BT);
    printf("Throughput: %.1f tok/s\n", BT / elapsed);
    
    // --------------------------------------------------
    // 6) Cleanup
    // --------------------------------------------------
    free(x);
    free(output);
    free(router_logits);
    free(entries);
    free(topk_indices);
    free(topk_weights);
    free(buf_gate);
    free(buf_up);
    free(buf_hidden);
    free(buf_expert_out);
    free(shared_gate);
    free(shared_up);
    free(shared_out);
    free(ffn_gate_inp);
    free(ffn_gate_inp_shexp);
    free(ffn_gate_exps);
    free(ffn_up_exps);
    free(ffn_down_exps);
    free(ffn_gate_shexp);
    free(ffn_up_shexp);
    free(ffn_down_shexp);
    gguf_close(ctx);
    
    printf("\nDone.\n");
    return 0;
}
