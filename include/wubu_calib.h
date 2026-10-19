/*
 * wubu_calib.h -- camera geometry: PnP by normalised DLT.
 *
 * Solves for the 3x4 projection matrix from >=6 3D<->2D correspondences with
 * p34 fixed to 1, using Hartley normalisation on both sides. This recovers the
 * camera matrix including intrinsics; it does not split K/R/t.
 */
#ifndef WUBU_CALIB_H
#define WUBU_CALIB_H

/* p3: n points as x,y,z triples. p2: n points as u,v pairs.
 * P: 12 doubles, row-major 3x4. Returns 0, or -1. */
int wubu_pnp_dlt(const double *p3, const double *p2, int n, double P[12]);

/* Root-mean-square reprojection error of P over the same points. */
double wubu_pnp_rmse(const double *p3, const double *p2, int n, const double P[12]);

/* Split P = K [R|t] by RQ decomposition of its 3x3 left block.
 * K: 9 doubles row-major upper-triangular, normalised so K[2][2] == 1.
 * R: 9 doubles row-major rotation, t: 3 doubles. Returns 0, or -1. */
int wubu_pnp_decompose(const double P[12], double K[9], double R[9], double t[3]);

#include "wubu_cv.h"

/* Homography from four point pairs (src -> dst), given as x0,y0,...,x3,y3.
 * H is row-major 3x3 with H[8] == 1. Returns 0, or -1. */
int wubu_homography(const double src[8], const double dst[8], double H[9]);

/* Sample src through H (src -> dst) into a new ow x oh image. */
int wubu_warp_perspective(const wubu_img *src, const double H[9],
                          wubu_img *out, int ow, int oh);

typedef struct {
    int id;              /* dictionary index, or -1 */
    int hamming;         /* best Hamming distance found */
    int bits;            /* decoded 16-bit payload */
} wubu_aruco_marker;

/* Decode a 4x4-payload marker (6x6 cells with a one-cell black border) from a
 * quadrilateral in `im`. Samples cell centres after warping, then matches the
 * 16-bit code against `dict` within `max_hamming`. Returns 0 on a match, -1 if
 * nothing matched or the quad is degenerate. */
int wubu_aruco_decode(const wubu_img *im, const double quad[8],
                      const uint16_t *dict, int ndict, int max_hamming,
                      wubu_aruco_marker *out);

#endif /* WUBU_CALIB_H */
