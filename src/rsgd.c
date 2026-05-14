/**
 * rsgd.c — Riemannian SGD optimizer for Poincaré ball parameters
 *
 * RSGD allows gradient-based optimization of parameters constrained to
 * the Poincaré ball. The key steps:
 *
 *   1. g = Euclidean gradient dL/dw (computed by standard backprop)
 *   2. g_riem = g * ((1 - ||w||²)² / 4)     (project to tangent space at w)
 *   3. delta = -lr * g_riem                    (step in tangent space)
 *   4. w_new = exp_map(w + delta, R)           (Euclidean step + exp_map to ball)
 *
 * Reference: "Riemannian Adaptive Optimization Methods" (Becigneul & Ganea, ICLR 2019)
 *            g_riem = λ_w² * g_euc where λ_w = 2/(1 - ||w||²) for Poincaré ball
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Apply one RSGD step to a batch of vectors in the Poincaré ball.
 *
 * w:        [n_vecs, dim]  — current parameter vectors (in ball), UPDATED IN-PLACE
 * dw:       [n_vecs, dim]  — Euclidean gradients (computed by standard backprop)
 * n_vecs:   number of vectors
 * dim:      dimension of each vector
 * lr:       learning rate
 * R:        Poincaré ball radius
 * clip:     gradient clipping factor (1.0 = no clip)
 *
 * For each vector w_i:
 *   1. Compute g_riem = dw_i * ((1 - ||w_i||²)² / 4)
 *   2. Step: delta = -lr * g_riem * clip
 *   3. Euclidean step: w_tmp = w_i + delta
 *   4. Project: w_i = exp_map(w_tmp, dim, R, w_i)
 */
void rsgd_step(float *w, const float *dw, int n_vecs, int dim,
               float lr, float R, float clip) {
    float *tmp = (float *)malloc(dim * sizeof(float));
    if (!tmp) { fprintf(stderr, "RSGD: allocation failed\n"); return; }
    
    for (int v = 0; v < n_vecs; v++) {
        float *w_v = w + v * dim;
        const float *dw_v = dw + v * dim;
        
        // Compute ||w_v||²
        double n2 = 0.0;
        for (int i = 0; i < dim; i++)
            n2 += (double)w_v[i] * (double)w_v[i];
        
        // Riemannian metric factor: λ_w² = ((1 - ||w||²)² / 4)
        double lambda2 = (1.0 - n2) * (1.0 - n2) / 4.0;
        
        // Step in tangent space: delta = -lr * g_riem = -lr * dw * λ_w²
        // Then Euclidean step: w_tmp = w + delta
        for (int i = 0; i < dim; i++) {
            double g_riem = (double)dw_v[i] * lambda2;
            double step = (double)(-lr) * g_riem * (double)clip;
            tmp[i] = (float)((double)w_v[i] + step);
        }
        
        // Project back to Poincaré ball via exp_map
        wubu_exp_map(tmp, dim, R, w_v);
    }
    
    free(tmp);
}
