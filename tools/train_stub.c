/**
 * train_stub.c — Phase 3 Training Loop Verification
 *
 * A self-contained training stub that:
 * 1. Creates a tiny random model (2 layers, small hidden dim)
 * 2. Generates synthetic data (random token sequences)
 * 3. Runs forward → cross-entropy loss → gradient (finite diff) → AdamW update
 * 4. Verifies loss decreases over training steps
 *
 * This verifies the training machinery works BEFORE we try it on
 * the full 40-layer Qwen3.6 model (which needs automatic differentiation).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ================================================================
// Tiny model configuration
// ================================================================
#define STUB_D_MODEL  16   // tiny hidden dimension
#define STUB_VOCAB    32   // small vocab
#define STUB_N_LAYERS 1    // 1 tiny layer

// ================================================================
// Timing
// ================================================================
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ================================================================
// Tiny model: 2-layer MLP with residual
// Structure: input -> linear(64 -> 64) -> ReLU -> linear(64 -> 64) -> residual
// Then: final_linear(64 -> vocab) -> logits
// ================================================================
typedef struct {
    // Layer 0
    float *w0, *b0;  // [D_MODEL, D_MODEL], [D_MODEL]
    // Layer 1
    float *w1, *b1;  // [D_MODEL, D_MODEL], [D_MODEL]
    // Output projection
    float *w_out, *b_out;  // [D_MODEL, VOCAB], [VOCAB]
    
    // Storage for parameters in contiguous array (for AdamW)
    float *params;
    int n_params;
} stub_model_t;

static void stub_init(stub_model_t *m) {
    memset(m, 0, sizeof(*m));
    
    int d = STUB_D_MODEL;
    int v = STUB_VOCAB;
    
    // Count total params
    m->n_params = 0;
    m->n_params += d*d + d;           // layer 0
    m->n_params += d*d + d;           // layer 1
    m->n_params += d*v + v;           // output
    // 2 layers: 2*(4096+64) + 16384+256 = 24896
    
    m->params = (float *)calloc(m->n_params, sizeof(float));
    int offset = 0;
    
    m->w0 = m->params + offset; offset += d*d;
    m->b0 = m->params + offset; offset += d;
    m->w1 = m->params + offset; offset += d*d;
    m->b1 = m->params + offset; offset += d;
    m->w_out = m->params + offset; offset += d*v;
    m->b_out = m->params + offset; offset += v;
    
    // Initialize with small random weights
    srand(42);
    for (int i = 0; i < m->n_params; i++) {
        m->params[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
}

static void stub_forward(stub_model_t *m, const float *x, int B, int T, float *logits) {
    int N = B * T;
    int d = STUB_D_MODEL;
    int v = STUB_VOCAB;
    
    // Layer 0: h = ReLU(x @ w0 + b0)
    // h = x @ w0
    float *h = (float *)calloc(N * d, sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            float sum = 0;
            for (int k = 0; k < d; k++) {
                sum += x[i * d + k] * m->w0[k * d + j];
            }
            h[i * d + j] = sum + m->b0[j];
            // ReLU
            if (h[i * d + j] < 0) h[i * d + j] = 0;
        }
    }
    
    // Layer 1: h2 = ReLU(h @ w1 + b1) + residual
    float *h2 = (float *)calloc(N * d, sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            float sum = 0;
            for (int k = 0; k < d; k++) {
                sum += h[i * d + k] * m->w1[k * d + j];
            }
            h2[i * d + j] = sum + m->b1[j];
            // ReLU
            if (h2[i * d + j] < 0) h2[i * d + j] = 0;
            // Residual
            h2[i * d + j] += h[i * d + j];
        }
    }
    free(h);
    
    // Output projection: logits = h2 @ w_out + b_out
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < v; j++) {
            float sum = 0;
            for (int k = 0; k < d; k++) {
                sum += h2[i * d + k] * m->w_out[k * v + j];
            }
            logits[i * v + j] = sum + m->b_out[j];
        }
    }
    free(h2);
}

// ================================================================
// Cross-entropy loss with softmax
// ================================================================
static float cross_entropy_loss(const float *logits, const int *targets,
                                 int N, int vocab_size, int *correct_out) {
    float total_loss = 0.0f;
    int correct = 0;
    
    for (int i = 0; i < N; i++) {
        // Find max for numerical stability
        float max_l = logits[i * vocab_size];
        for (int j = 1; j < vocab_size; j++) {
            if (logits[i * vocab_size + j] > max_l)
                max_l = logits[i * vocab_size + j];
        }
        
        // Softmax
        float sum_exp = 0.0f;
        for (int j = 0; j < vocab_size; j++) {
            sum_exp += expf(logits[i * vocab_size + j] - max_l);
        }
        
        // Loss = -log(softmax[target])
        int t = targets[i];
        float softmax_t = expf(logits[i * vocab_size + t] - max_l) / sum_exp;
        total_loss += -logf(softmax_t + 1e-30f);
        
        // Check accuracy
        int pred = 0;
        float max_p = logits[i * vocab_size];
        for (int j = 1; j < vocab_size; j++) {
            if (logits[i * vocab_size + j] > max_p) {
                max_p = logits[i * vocab_size + j];
                pred = j;
            }
        }
        if (pred == t) correct++;
    }
    
    if (correct_out) *correct_out = correct;
    return total_loss / N;
}

// ================================================================
// Finite-difference gradients (for verification)
// ================================================================
static float compute_loss_for_params(stub_model_t *m, const float *x, 
                                      int B, int T, const int *targets) {
    int N = B * T;
    int v = STUB_VOCAB;
    float *logits = (float *)malloc(N * v * sizeof(float));
    stub_forward(m, x, B, T, logits);
    float loss = cross_entropy_loss(logits, targets, N, v, NULL);
    free(logits);
    return loss;
}

static void compute_gradients_fd(stub_model_t *m, const float *x,
                                  int B, int T, const int *targets,
                                  float *grad) {
    float eps = 1e-4f;
    float base_loss = compute_loss_for_params(m, x, B, T, targets);
    
    for (int i = 0; i < m->n_params; i++) {
        m->params[i] += eps;
        float loss_plus = compute_loss_for_params(m, x, B, T, targets);
        m->params[i] -= eps;
        grad[i] = (loss_plus - base_loss) / eps;
    }
}

// ================================================================
// AdamW Optimizer
// ================================================================
typedef struct {
    float *m;  // first moment
    float *v;  // second moment
    int t;     // timestep
    float lr, beta1, beta2, eps, wd;
} adamw_t;

static void adamw_init(adamw_t *opt, int n_params) {
    opt->m = (float *)calloc(n_params, sizeof(float));
    opt->v = (float *)calloc(n_params, sizeof(float));
    opt->t = 0;
    opt->lr = 1e-3f;
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->eps = 1e-8f;
    opt->wd = 0.01f;
}

static void adamw_step(adamw_t *opt, float *params, const float *grad, int n) {
    opt->t++;
    float b1 = opt->beta1;
    float b2 = opt->beta2;
    float lr = opt->lr;
    float eps = opt->eps;
    float wd = opt->wd;
    
    float bias_corr1 = 1.0f - powf(b1, opt->t);
    float bias_corr2 = 1.0f - powf(b2, opt->t);
    
    for (int i = 0; i < n; i++) {
        // Weight decay
        params[i] -= lr * wd * params[i];
        
        // Update moments
        opt->m[i] = b1 * opt->m[i] + (1 - b1) * grad[i];
        opt->v[i] = b2 * opt->v[i] + (1 - b2) * grad[i] * grad[i];
        
        // Bias-corrected moments
        float m_hat = opt->m[i] / bias_corr1;
        float v_hat = opt->v[i] / bias_corr2;
        
        // Update
        params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

static void adamw_free(adamw_t *opt) {
    free(opt->m);
    free(opt->v);
}

// ================================================================
// Data helpers: generate random token sequences
// ================================================================
static void generate_data(int *tokens, int B, int T, int vocab) {
    srand(123);
    for (int i = 0; i < B * T; i++) {
        tokens[i] = rand() % vocab;
    }
}

// Simple embedding: one-hot-like embedding matrix
static void embed_tokens(const int *tokens, float *embeddings, int B, int T, int d, int vocab) {
    srand(456);
    for (int i = 0; i < B * T; i++) {
        int id = tokens[i] % vocab;
        // Use random embedding for each token (fixed)
        for (int j = 0; j < d; j++) {
            // Deterministic embedding based on token id
            embeddings[i * d + j] = sinf((float)(id * (j+1))) * 0.5f;
        }
    }
}

// ================================================================
// Main: training loop
// ================================================================
int main() {
    printf("========================================================\n");
    printf("  WuBuText AI — Phase 3 Training Loop Stub\n");
    printf("========================================================\n");
    printf("  D_MODEL=%d, VOCAB=%d, Layers=%d\n", STUB_D_MODEL, STUB_VOCAB, STUB_N_LAYERS);
    printf("  Optimizer: AdamW (lr=%.0e, wd=%.2f)\n", 1e-3f, 0.01f);
    printf("  Gradients: Finite differences (eps=1e-4)\n");
    
    // Init model
    stub_model_t model;
    stub_init(&model);
    printf("\n  Parameters: %d\n", model.n_params);
    
    // Init optimizer
    adamw_t opt;
    adamw_init(&opt, model.n_params);
    
    // Training config
    int B = 4, T = 16;
    int N = B * T;
    int n_steps = 50;
    int eval_every = 10;
    
    printf("  Batch: B=%d, T=%d\n", B, T);
    printf("  Steps: %d\n\n", n_steps);
    
    // Generate fixed training data
    int *train_tokens = (int *)malloc(N * sizeof(int));
    float *train_embd = (float *)malloc(N * STUB_D_MODEL * sizeof(float));
    generate_data(train_tokens, B, T, STUB_VOCAB);
    embed_tokens(train_tokens, train_embd, B, T, STUB_D_MODEL, STUB_VOCAB);
    
    // Targets: predict next token (shift by 1)
    int *targets = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N - 1; i++) targets[i] = train_tokens[i + 1];
    targets[N - 1] = 0;  // pad
    
    // Gradient buffer
    float *grad = (float *)malloc(model.n_params * sizeof(float));
    float *logits = (float *)malloc(N * STUB_VOCAB * sizeof(float));
    
    printf("  Initial loss: ");
    int init_correct;
    float initial_loss = cross_entropy_loss(logits, targets, N, STUB_VOCAB, &init_correct);
    printf("%.6f  acc = %.1f%%\n", initial_loss, (float)init_correct / N * 100.0f);
    
    // Training loop
    double t0 = now_sec();
    float prev_loss = initial_loss;
    
    for (int step = 0; step < n_steps; step++) {
        // Forward
        stub_forward(&model, train_embd, B, T, logits);
        int correct;
        float loss = cross_entropy_loss(logits, targets, N, STUB_VOCAB, &correct);
        
        // Backward (finite differences)
        compute_gradients_fd(&model, train_embd, B, T, targets, grad);
        
        // Update
        adamw_step(&opt, model.params, grad, model.n_params);
        
        if ((step + 1) % eval_every == 0) {
            float acc = (float)correct / N * 100.0f;
            printf("  Step %4d: loss = %.6f  acc = %.1f%%  (delta = %+.6f)\n",
                   step + 1, loss, acc, prev_loss - loss);
            prev_loss = loss;
        }
    }
    
    double elapsed = now_sec() - t0;
    
    printf("\n========================================================\n");
    printf("  RESULTS\n");
    printf("========================================================\n");
    printf("  Final loss: %.6f\n", prev_loss);
    printf("  Initial loss: %.6f\n", initial_loss);
    printf("  Delta: %+.6f (%.1f%%)\n", 
           initial_loss - prev_loss,
           (initial_loss - prev_loss) / initial_loss * 100);
    printf("  Time: %.2f s (%.2f ms/step)\n", elapsed, elapsed / n_steps * 1000);
    
    if (prev_loss < initial_loss * 0.9f) {
        printf("  PASS: Loss decreased >10%% — training works!\n");
    } else if (prev_loss < initial_loss) {
        printf("  WARN: Loss decreased but <10%% — gradient may need tuning\n");
    } else {
        printf("  FAIL: Loss did not decrease — check gradient computation\n");
    }
    
    // Cleanup
    free(train_tokens);
    free(train_embd);
    free(targets);
    free(grad);
    free(logits);
    adamw_free(&opt);
    free(model.params);
    
    return 0;
}
