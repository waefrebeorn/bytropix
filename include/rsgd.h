#ifndef WUBU_RSGD_H
#define WUBU_RSGD_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Apply one RSGD step to a batch of vectors in the Poincaré ball.
 *
 * w:        [n_vecs, dim]  — current param vectors (in ball), UPDATED IN-PLACE
 * dw:       [n_vecs, dim]  — Euclidean gradients
 * n_vecs:   number of vectors
 * dim:      dimension of each vector
 * lr:       learning rate
 * R:        Poincaré ball radius
 * clip:     gradient clipping factor (1.0 = no clip)
 */
void rsgd_step(float *w, const float *dw, int n_vecs, int dim,
               float lr, float R, float clip);

#ifdef __cplusplus
}
#endif

#endif // WUBU_RSGD_H
