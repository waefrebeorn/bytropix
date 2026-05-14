/**
 * nn_ops.h — Neural Network Ops (pure C, single-precision)
 *
 * Forward and backward operations for transformer building blocks.
 * All operations work on flat float arrays for simplicity.
 */
#ifndef NN_OPS_H
#define NN_OPS_H

#include <math.h>
#include <stdint.h>
#include <string.h>

/* ─── Math Utilities ─── */
static inline float nn_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static inline float nn_gelu_deriv(float x) {
    float t = tanhf(0.7978845608f * (x + 0.044715f * x * x * x));
    float inner_grad = 0.7978845608f * (1.0f + 0.134145f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * inner_grad;
}

static inline float nn_relu(float x) { return x > 0.0f ? x : 0.0f; }
static inline float nn_relu_deriv(float x) { return x > 0.0f ? 1.0f : 0.0f; }

static inline float nn_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* Softmax in-place */
static inline void nn_softmax(float* x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    float inv = 1.0f / (sum + 1e-7f);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* Cross-entropy loss: loss = -log(p[target]) where p = softmax(logits) */
static inline float nn_cross_entropy_loss(const float* logits, int target, int n) {
    float maxv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(logits[i] - maxv);
    float log_sum = logf(sum + 1e-7f);
    return log_sum - (logits[target] - maxv);
}

/* Softmax derivative = p * (delta_ij - p_j) — returned as gradient w.r.t. logits */
static inline void nn_cross_entropy_grad(const float* logits, int target, int n,
                                          float* grad_out) {
    float maxv = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { grad_out[i] = expf(logits[i] - maxv); sum += grad_out[i]; }
    float inv = 1.0f / (sum + 1e-7f);
    for (int i = 0; i < n; i++) {
        grad_out[i] *= inv;                              /* softmax prob */
        grad_out[i] -= (i == target) ? 1.0f : 0.0f;      /* grad = p - target_onehot */
    }
}

/* ─── Linear Algebra (hand-rolled, single-threaded) ─── */
/* y = A^T * x  (A is n x m stored row-major) */
static inline void nn_matmul_vec(const float* A, const float* x, float* y,
                                  int n, int m) {
    for (int j = 0; j < m; j++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++)
            sum += A[i * m + j] * x[i];
        y[j] = sum;
    }
}

/* y += A^T * x */
static inline void nn_matmul_vec_add(const float* A, const float* x, float* y,
                                      int n, int m) {
    for (int j = 0; j < m; j++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++)
            sum += A[i * m + j] * x[i];
        y[j] += sum;
    }
}

/* y = x^T * A  (row vector x * matrix A -- produces row vector y of length m) */
static inline void nn_vec_matmul(const float* x, const float* A, float* y,
                                  int n, int m) {
    for (int j = 0; j < m; j++) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++)
            sum += x[i] * A[i * m + j];
        y[j] = sum;
    }
}

/* ─── Layer Normalization ─── */
static inline float nn_layer_norm(float x, float mean, float var,
                                   float gamma, float beta) {
    return gamma * (x - mean) / sqrtf(var + 1e-5f) + beta;
}

static inline void nn_layer_norm_forward(const float* x, float* y,
                                          int n, const float* gamma, const float* beta) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= n;
    for (int i = 0; i < n; i++)
        y[i] = gamma[i] * (x[i] - mean) / sqrtf(var + 1e-5f) + beta[i];
}

/* Backward through layer norm: given dL/dy, compute dL/dx, dL/dgamma, dL/dbeta */
static inline void nn_layer_norm_backward(const float* x, const float* dy,
                                           float* dx, float* dgamma, float* dbeta,
                                           int n, const float* gamma) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= n;
    float std = sqrtf(var + 1e-5f);
    float inv_std = 1.0f / std;

    /* dL/dbeta = sum(dy) */
    for (int i = 0; i < n; i++) dbeta[i] = dy[i];
    /* dL/dgamma = sum(dy * (x-mean)/std) */
    for (int i = 0; i < n; i++) dgamma[i] = dy[i] * (x[i] - mean) * inv_std;

    /* dL/dx = (gamma/n) * (n*dy - sum(dy) - (x-mean)/std * sum(dy*(x-mean)/std)) / std */
    float sum_dy = 0.0f, sum_dy_norm = 0.0f;
    for (int i = 0; i < n; i++) { sum_dy += dy[i]; sum_dy_norm += dy[i] * (x[i] - mean); }
    float inv_n = 1.0f / n;
    for (int i = 0; i < n; i++) {
        float grad = (n * dy[i] - sum_dy - (x[i] - mean) * inv_std * inv_std * sum_dy_norm) * inv_n;
        dx[i] = gamma[i] * grad * inv_std;
    }
}

#endif /* NN_OPS_H */
