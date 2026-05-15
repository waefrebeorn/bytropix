#include "wubu_hyperbolic_output_proj.h"
#include "wubu_mobius_linear.h"
#include "gguf_reader.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================
// Helper: exp_map backward
// exp_map(v): output[i] = tanh(||v||/R) * R/||v|| * v[i]
// Backprop: dL/dv = ...
// ============================================================
static void exp_map_backward(const float *v, int d, float R,
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
// Backprop: dL/dx = ...
// ============================================================
static void log_map_backward(const float *x, int d, float R,
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
    float nx2 = nx * nx;
    float denom = R2 - nx2;
    if (denom < 1e-12f) denom = 1e-12f;
    float fp_num = R2 * nx / denom - R * atanh_r;
    float fp = fp_num / nx2;

    float dot = 0.0f;
    for (int i = 0; i < d; i++) dot += d_output[i] * x[i];

    float factor = fp / nx;
    for (int i = 0; i < d; i++) {
        d_input[i] = d_output[i] * f + factor * x[i] * dot;
    }
}

// ============================================================
// Forward
// ============================================================
void wubu_hyperbolic_output_proj_forward(
    const float *hidden, int N, int D, int V,
    const float *W, const float *b,
    float R_hidden, float R_logit,
    float *logits,
    float *h_ball,
    float *l_ball)
{
    // Allocate buffers for intermediates (or use provided storage)
    int need_free_h_ball = 0;
    int need_free_l_ball = 0;

    if (!h_ball) {
        h_ball = (float *)malloc((int64_t)N * D * sizeof(float));
        if (!h_ball) { fprintf(stderr, "HypOutProj fwd: h_ball alloc failed\n"); return; }
        need_free_h_ball = 1;
    }
    if (!l_ball) {
        l_ball = (float *)malloc((int64_t)N * V * sizeof(float));
        if (!l_ball) {
            fprintf(stderr, "HypOutProj fwd: l_ball alloc failed\n");
            if (need_free_h_ball) free((float *)h_ball);
            return;
        }
        need_free_l_ball = 1;
    }

    // Step 1: hidden (Euclidean) → h_ball (Poincaré ball of radius R_hidden)
    for (int i = 0; i < N; i++) {
        wubu_exp_map(hidden + (int64_t)i * D, D, R_hidden, h_ball + (int64_t)i * D);
    }

    // Step 2: M⊗(h_ball, R_hidden → R_logit)
    //   = exp_map(W @ log_map(h_ball, R_hidden) + b, R_logit)
    // This uses wubu_mobius_linear_forward which does exactly this.
    wubu_mobius_linear_forward(h_ball, N, D, V, W, b, R_hidden, R_logit, l_ball);

    // Step 3: l_ball (Poincaré ball of radius R_logit) → logits (Euclidean)
    for (int i = 0; i < N; i++) {
        wubu_log_map(l_ball + (int64_t)i * V, V, R_logit, logits + (int64_t)i * V);
    }

    if (need_free_h_ball) free(h_ball);
    if (need_free_l_ball) free(l_ball);
}

// ============================================================
// Backward
// ============================================================
void wubu_hyperbolic_output_proj_backward(
    const float *hidden, int N, int D, int V,
    const float *W, const float *b,
    float R_hidden, float R_logit,
    const float *h_ball,
    const float *l_ball,
    const float *d_logits,
    float *d_hidden,
    float *d_W, float *d_b)
{
    // ----- Allocate/recompute intermediates -----

    // h_ball: needed for mobius_linear_backward (input x).
    // If not saved, recompute from hidden.
    float *h_ball_buf = NULL;
    int need_free_h = 0;
    if (h_ball) {
        h_ball_buf = (float *)h_ball; // const-cast, won't write
    } else {
        h_ball_buf = (float *)malloc((int64_t)N * D * sizeof(float));
        if (!h_ball_buf) { fprintf(stderr, "HypOutProj bwd: h_ball alloc failed\n"); return; }
        need_free_h = 1;
        for (int i = 0; i < N; i++) {
            wubu_exp_map(hidden + (int64_t)i * D, D, R_hidden, h_ball_buf + (int64_t)i * D);
        }
    }

    // l_ball: needed for log_map backward and mobius_linear_backward output.
    // If not saved, recompute by running M⊗ forward on h_ball.
    // Note: This doubles the forward compute cost but avoids N×V storage.
    float *l_ball_buf = NULL;
    int need_free_l = 0;
    if (l_ball) {
        l_ball_buf = (float *)l_ball;
    } else {
        l_ball_buf = (float *)malloc((int64_t)N * V * sizeof(float));
        if (!l_ball_buf) {
            fprintf(stderr, "HypOutProj bwd: l_ball alloc failed\n");
            if (need_free_h) free(h_ball_buf);
            return;
        }
        need_free_l = 1;
        wubu_mobius_linear_forward(h_ball_buf, N, D, V, W, b, R_hidden, R_logit, l_ball_buf);
    }

    // ----- Allocate gradient buffers -----

    // d_l_ball: gradient w.r.t. l_ball (output of M⊗, input to log_map)
    float *d_l_ball = (float *)malloc((int64_t)N * V * sizeof(float));
    if (!d_l_ball) {
        fprintf(stderr, "HypOutProj bwd: d_l_ball alloc failed\n");
        if (need_free_h) free(h_ball_buf);
        if (need_free_l) free(l_ball_buf);
        return;
    }

    // d_h_ball: gradient w.r.t. h_ball (output of exp_map, input to M⊗)
    float *d_h_ball = (float *)calloc((int64_t)N * D, sizeof(float));
    if (!d_h_ball) {
        fprintf(stderr, "HypOutProj bwd: d_h_ball alloc failed\n");
        free(d_l_ball);
        if (need_free_h) free(h_ball_buf);
        if (need_free_l) free(l_ball_buf);
        return;
    }

    // ----- Step 1: Backprop through log_map (l_ball → logits) -----
    // d_logits is gradient w.r.t. Euclidean logits
    // d_l_ball = d_logits * d(log_map)/d(l_ball)
    for (int i = 0; i < N; i++) {
        log_map_backward(l_ball_buf + (int64_t)i * V, V, R_logit,
                         d_logits + (int64_t)i * V,
                         d_l_ball + (int64_t)i * V);
    }

    // ----- Step 2: Backprop through M⊗ (h_ball → l_ball) -----
    // This gives us d_h_ball, d_W, d_b from d_l_ball
    wubu_mobius_linear_backward(
        h_ball_buf, N, D, V,
        NULL,     // tangent_x: let backward recompute from h_ball
        NULL,     // tangent_out: let backward recompute from W and tangent_x
        l_ball_buf,
        d_l_ball,
        W,
        R_hidden, R_logit,
        d_h_ball,
        d_W, d_b);

    // ----- Step 3: Backprop through exp_map (hidden → h_ball) -----
    // d_hidden = d_h_ball * d(exp_map)/d(hidden)
    if (d_hidden) {
        for (int i = 0; i < N; i++) {
            float *dh_s = d_hidden + (int64_t)i * D;
            float *d_hb_s = d_h_ball + (int64_t)i * D;
            // We need to ADD to existing d_hidden, so use temp buffer and add
            float *tmp_dh = (float *)malloc(D * sizeof(float));
            if (!tmp_dh) {
                fprintf(stderr, "HypOutProj bwd: tmp_dh alloc failed\n");
                // Graceful degradation: just add d_h_ball directly (approx)
                for (int j = 0; j < D; j++) dh_s[j] += d_hb_s[j];
                free(d_l_ball); free(d_h_ball);
                if (need_free_h) free(h_ball_buf);
                if (need_free_l) free(l_ball_buf);
                return;
            }
            exp_map_backward(hidden + (int64_t)i * D, D, R_hidden,
                             d_hb_s, tmp_dh);
            for (int j = 0; j < D; j++) dh_s[j] += tmp_dh[j];
            free(tmp_dh);
        }
    }

    // ----- Cleanup -----
    free(d_l_ball);
    free(d_h_ball);
    if (need_free_h) free(h_ball_buf);
    if (need_free_l) free(l_ball_buf);
}
