#include "wubu_mobius_linear.h"
#include "wubu_mobius.h"
#include "gguf_reader.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================
// Helper: exp_map backward (matching interface from PGA backend)
// exp_map(v): output[i] = tanh(||v||/R) * R/||v|| * v[i]
// ============================================================
static void mobius_linear_exp_map_backward(const float *v, int d, float R,
                                            const float *d_output, float *d_input) {
    float nv = 0.0f;
    for (int i = 0; i < d; i++) nv += v[i] * v[i];
    nv = sqrtf(nv);
    if (nv < 1e-12f) {
        memcpy(d_input, d_output, d * sizeof(float));
        return;
    }
    float ratio = nv / R;
    if (ratio > 0.99f) ratio = 0.99f;
    float th = tanhf(ratio);
    float g = th * R / nv;
    float sech2 = 1.0f - th * th;
    float gp = (sech2 * nv - th * R) / (nv * nv);

    float dot = 0.0f;
    for (int i = 0; i < d; i++) dot += d_output[i] * v[i];

    float factor = gp / nv;
    for (int i = 0; i < d; i++) {
        d_input[i] = d_output[i] * g + factor * v[i] * dot;
    }
}

// ============================================================
// Helper: log_map backward
// log_map(x): output[i] = R * atanh(||x||/R) / ||x|| * x[i]
// ============================================================
static void mobius_linear_log_map_backward(const float *x, int d, float R,
                                            const float *d_output, float *d_input) {
    float nx = 0.0f;
    for (int i = 0; i < d; i++) nx += x[i] * x[i];
    nx = sqrtf(nx);
    if (nx < 1e-12f) {
        memcpy(d_input, d_output, d * sizeof(float));
        return;
    }
    float ratio = nx / R;
    if (ratio > 0.999f) ratio = 0.999f;
    float atanh_r = 0.5f * logf((1.0f + ratio) / (1.0f - ratio));
    float f = R * atanh_r / nx;

    float R2 = R * R;
    float denom = R2 - nx * nx;
    if (denom < 1e-12f) denom = 1e-12f;
    float fp_num = R2 * nx / denom - R * atanh_r;
    float fp = fp_num / (nx * nx);

    float dot = 0.0f;
    for (int i = 0; i < d; i++) dot += d_output[i] * x[i];

    float factor = fp / nx;
    for (int i = 0; i < d; i++) {
        d_input[i] = d_output[i] * f + factor * x[i] * dot;
    }
}

// ============================================================
// Forward: y = exp_map(W @ log_map(x) + b, R_out)
// ============================================================
void wubu_mobius_linear_forward(const float *x, int N, int D_in, int D_out,
                                const float *W, const float *b,
                                float R_in, float R_out,
                                float *output) {
    // Per-sample temporary: log_map(x) then linear then exp_map
    float *log_x = (float *)malloc(D_in * sizeof(float));
    float *linear_out = (float *)malloc(D_out * sizeof(float));
    if (!log_x || !linear_out) {
        free(log_x); free(linear_out);
        fprintf(stderr, "MobiusLinear forward: alloc failed\n");
        return;
    }

    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_in;

        // Step 1: log_map(x, R_in) → tangent vector
        wubu_log_map(x_s, D_in, R_in, log_x);

        // Step 2: Euclidean linear: W @ log_x + b
        for (int j = 0; j < D_out; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_in; i++)
                sum += (double)W[j * D_in + i] * (double)log_x[i];
            if (b) sum += (double)b[j];
            linear_out[j] = (float)sum;
        }

        // Step 3: exp_map(linear_out, R_out) → Poincaré ball
        wubu_exp_map(linear_out, D_out, R_out, output + s * D_out);
    }

    free(log_x);
    free(linear_out);
}

// ============================================================
// Backward: gradients through M⊗ layer
// ============================================================
void wubu_mobius_linear_backward(const float *x, int N, int D_in, int D_out,
                                 const float *tangent_x,
                                 const float *tangent_out,
                                 const float *output,
                                 const float *d_output,
                                 const float *W,
                                 float R_in, float R_out,
                                 float *d_x,
                                 float *d_W, float *d_b) {
    // Per-sample temporaries
    float *tan_x = (float *)malloc(D_in * sizeof(float));
    float *tan_out = (float *)malloc(D_out * sizeof(float));
    float *d_tan_out = (float *)malloc(D_out * sizeof(float));
    float *d_tan_x = (float *)malloc(D_in * sizeof(float));
    if (!tan_x || !tan_out || !d_tan_out || !d_tan_x) {
        free(tan_x); free(tan_out); free(d_tan_out); free(d_tan_x);
        return;
    }

    // Zero weight gradient accumulators if needed
    if (d_W) memset(d_W, 0, (int64_t)D_out * D_in * sizeof(float));
    if (d_b) memset(d_b, 0, D_out * sizeof(float));

    for (int s = 0; s < N; s++) {
        const float *x_s = x + s * D_in;
        const float *dy_s = d_output + s * D_out;

        // Use saved tangent values if available, otherwise recompute
        if (tangent_x) {
            memcpy(tan_x, tangent_x + s * D_in, D_in * sizeof(float));
        } else {
            wubu_log_map(x_s, D_in, R_in, tan_x);
        }

        // Step 1: Backprop through exp_map
        // tangent_out = the pre-exp_map linear output
        if (tangent_out) {
            memcpy(tan_out, tangent_out + s * D_out, D_out * sizeof(float));
        } else {
            // Recompute: W @ tan_x + b
            for (int j = 0; j < D_out; j++) {
                double sum = 0.0;
                for (int i = 0; i < D_in; i++)
                    sum += (double)W[j * D_in + i] * (double)tan_x[i];
                tan_out[j] = (float)sum;
            }
        }

        // d_output → d_tan_out (backprop through exp_map)
        mobius_linear_exp_map_backward(tan_out, D_out, R_out, dy_s, d_tan_out);

        // Step 2: d_W = tan_x ⊗ d_tan_out  (outer product, accumulate)
        if (d_W) {
            for (int j = 0; j < D_out; j++) {
                float dt = d_tan_out[j];
                for (int i = 0; i < D_in; i++) {
                    d_W[j * D_in + i] += tan_x[i] * dt;
                }
            }
        }

        // d_b = Σ d_tan_out over batch
        if (d_b) {
            for (int j = 0; j < D_out; j++)
                d_b[j] += d_tan_out[j];
        }

        // Step 3: d_tan_x = W^T · d_tan_out  (backprop through matmul)
        memset(d_tan_x, 0, D_in * sizeof(float));
        for (int i = 0; i < D_in; i++) {
            double sum = 0.0;
            for (int j = 0; j < D_out; j++)
                sum += (double)W[j * D_in + i] * (double)d_tan_out[j];
            d_tan_x[i] = (float)sum;
        }

        // Step 4: Backprop through log_map to get d_x
        if (d_x) {
            float *dx_s = d_x + s * D_in;
            mobius_linear_log_map_backward(x_s, D_in, R_in, d_tan_x, dx_s);
        }
    }

    free(tan_x);
    free(tan_out);
    free(d_tan_out);
    free(d_tan_x);
}
