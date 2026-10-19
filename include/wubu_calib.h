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

/* Pose of a planar square marker from its four image corners (TL,TR,BR,BL).
 * Decomposes the homography: A = K^-1 H, lambda = 1/|a1|, r1 = lambda a1,
 * r2 = lambda a2, r3 = r1 x r2, t = lambda a3 (sign chosen so t_z > 0).
 * marker_size is the square's side in the same units as t. Returns 0, or -1. */
int wubu_marker_pose(const double corners[8], double marker_size,
                     const double K[9], double R[9], double t[3]);

typedef struct { double R[9]; double t[3]; double weight; } wubu_pose;

/* Fuse n poses: weighted mean translation, and a quaternion average of the
 * rotations (signs aligned to the first quaternion, then normalised).
 * Returns 0, or -1. */
int wubu_pose_fuse(const wubu_pose *in, int n, double R[9], double t[3]);

/* Gauss-Newton refinement of (R,t) against 3D-2D correspondences.
 * The rotation update is R <- exp([dw]x) R (first-order exponential);
 * the Jacobian is numeric. Returns 0, or -1. */
int wubu_pose_refine(const double *p3, const double *p2, int n, const double K[9],
                     double R[9], double t[3], int iters);

/* Refine one camera pose against several markers at once. marker_xyz holds
 * nmarkers*12 doubles (4 corners x 3D each); marker_uv holds nmarkers*8.
 * Returns 0, or -1. */
int wubu_markers_refine(const double *marker_xyz, const double *marker_uv,
                        int nmarkers, const double K[9],
                        double R[9], double t[3], int iters);

/* Recover a square-pixel focal length from correspondences: coarse log scan
 * over [0.3, 3] * K0[0], then golden-section refinement, solving the pose at
 * each trial. Updates K, R and t. Returns 0, or -1. */
int wubu_calibrate_focal(const double *p3, const double *p2, int n,
                         double K[9], double R[9], double t[3]);

/* Recover the principal point with the focal length held fixed: alternate
 * golden-section searches on cx and cy, refining the pose at each trial.
 * Returns 0, or -1. */
int wubu_calibrate_principal(const double *p3, const double *p2, int n,
                             double K[9], double R[9], double t[3]);

/* Radial distortion (Brown-Conrady, first two terms) around the principal point.
 * r2 is measured in normalised (focal-length) units. */
void wubu_distort_point(double x, double y, const double K[9], double k1, double k2,
                        double *ox, double *oy);

/* Inverse of wubu_distort_point by fixed-point iteration (20 steps).
 * Returns 0, or -1 when the sample does not converge. */
int wubu_undistort_point(double xd, double yd, const double K[9], double k1, double k2,
                         double *ox, double *oy);

#endif /* WUBU_CALIB_H */
