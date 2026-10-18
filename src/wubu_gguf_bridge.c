#include "gguf_reader.h"
#include "wubu_gguf_tensor.h"
#include <math.h>
#include <string.h>

// ========== Poincare / Hyperbolic Ops ==========

float wubu_norm(const float *v, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) sum += (double)v[i] * v[i];
    return (float)sqrt(sum);
}

float wubu_dot(const float *a, const float *b, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) sum += (double)a[i] * b[i];
    return (float)sum;
}

/* wubu_exp_map / wubu_log_map now live in wubu_mobius.c (the canonical
 * home declared by wubu_mobius.h). The identical copies that used to sit
 * here were a latent duplicate-symbol bug. */