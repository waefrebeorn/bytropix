/* wubu_muon.h -- Newton-Schulz polar orthogonalization (public primitive). */
#ifndef WUBU_MUON_H
#define WUBU_MUON_H

/* Orthogonalize x[rows*cols] in place: after convergence X^T X = I (tall) or
 * X X^T = I (wide, the transpose path). Returns 0, or -1 on bad arguments.
 * The matrix is Frobenius-normalized once, then refined for `iters` steps. */
int wubu_muon_orthogonalize(float *x, int rows, int cols, int iters);

#endif /* WUBU_MUON_H */