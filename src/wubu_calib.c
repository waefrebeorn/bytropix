/*
 * wubu_calib.c -- DLT PnP. See wubu_calib.h.
 */
#include "wubu_calib.h"
#include <math.h>
#include <string.h>

/* Solve A x = b for an 11x11 system (partial pivoting). Returns 0 on success. */
static int solve11(double A[11][11], double b[11], double x[11])
{
    for (int col = 0; col < 11; col++) {
        int piv = col;
        for (int r = col + 1; r < 11; r++)
            if (fabs(A[r][col]) > fabs(A[piv][col])) piv = r;
        if (fabs(A[piv][col]) < 1e-12) return -1;

        if (piv != col) {
            for (int k = 0; k < 11; k++) {
                double t = A[col][k]; A[col][k] = A[piv][k]; A[piv][k] = t;
            }
            double t = b[col]; b[col] = b[piv]; b[piv] = t;
        }
        for (int r = col + 1; r < 11; r++) {
            double f = A[r][col] / A[col][col];
            if (f == 0.0) continue;
            for (int k = col; k < 11; k++) A[r][k] -= f * A[col][k];
            b[r] -= f * b[col];
        }
    }
    for (int i = 10; i >= 0; i--) {
        double s = b[i];
        for (int k = i + 1; k < 11; k++) s -= A[i][k] * x[k];
        if (fabs(A[i][i]) < 1e-12) return -1;
        x[i] = s / A[i][i];
    }
    return 0;
}

int wubu_pnp_dlt(const double *p3, const double *p2, int n, double P[12])
{
    if (!p3 || !p2 || !P || n < 6) return -1;

    /* Hartley normalisation: centre on the centroid, scale by mean distance. */
    double c3[3] = {0, 0, 0}, c2[2] = {0, 0};
    for (int i = 0; i < n; i++) {
        c3[0] += p3[3 * i]; c3[1] += p3[3 * i + 1]; c3[2] += p3[3 * i + 2];
        c2[0] += p2[2 * i]; c2[1] += p2[2 * i + 1];
    }
    for (int k = 0; k < 3; k++) c3[k] /= n;
    for (int k = 0; k < 2; k++) c2[k] /= n;

    double d3 = 0.0, d2 = 0.0;
    for (int i = 0; i < n; i++) {
        double x = p3[3 * i] - c3[0], y = p3[3 * i + 1] - c3[1], z = p3[3 * i + 2] - c3[2];
        d3 += sqrt(x * x + y * y + z * z);
        d2 += hypot(p2[2 * i] - c2[0], p2[2 * i + 1] - c2[1]);
    }
    d3 /= n; d2 /= n;
    if (d3 < 1e-12 || d2 < 1e-12) return -1;
    double s3 = sqrt(3.0) / d3, s2 = sqrt(2.0) / d2;

    double A[11][11];
    double b[11];
    memset(A, 0, sizeof A);
    memset(b, 0, sizeof b);

    for (int i = 0; i < n; i++) {
        double X = (p3[3 * i] - c3[0]) * s3;
        double Y = (p3[3 * i + 1] - c3[1]) * s3;
        double Z = (p3[3 * i + 2] - c3[2]) * s3;
        double u = (p2[2 * i] - c2[0]) * s2;
        double v = (p2[2 * i + 1] - c2[1]) * s2;

        double r1[11] = {X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z};
        double r2[11] = {0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z};

        for (int a = 0; a < 11; a++) {
            for (int c = 0; c < 11; c++) A[a][c] += r1[a] * r1[c] + r2[a] * r2[c];
            b[a] += r1[a] * u + r2[a] * v;
        }
    }

    double x[11];
    if (solve11(A, b, x) != 0) return -1;

    /* Pn maps normalised 3D to normalised 2D; undo both normalisations. */
    double Pn[12] = {x[0], x[1], x[2], x[3],
                     x[4], x[5], x[6], x[7],
                     x[8], x[9], x[10], 1.0};

    /* T3: (X - c3)*s3  =>  scale rows then subtract. T2^-1: u/s2 + c2. */
    double M[12];
    for (int r = 0; r < 3; r++) {
        M[4 * r + 0] = Pn[4 * r + 0] * s3;
        M[4 * r + 1] = Pn[4 * r + 1] * s3;
        M[4 * r + 2] = Pn[4 * r + 2] * s3;
        M[4 * r + 3] = Pn[4 * r + 3] - s3 * (Pn[4 * r + 0] * c3[0]
                                            + Pn[4 * r + 1] * c3[1]
                                            + Pn[4 * r + 2] * c3[2]);
    }
    /* P = T2inv * M, with T2inv = [[1/s2,0,c2u],[0,1/s2,c2v],[0,0,1]].
     * Homogeneous scale is free, so rows 0/1 carry 1/s2 and the centroid term
     * comes from row 2 unscaled. */
    for (int c = 0; c < 4; c++) {
        P[c]     = M[c] / s2 + c2[0] * M[8 + c];
        P[4 + c] = M[4 + c] / s2 + c2[1] * M[8 + c];
        P[8 + c] = M[8 + c];
    }
    return 0;
}

double wubu_pnp_rmse(const double *p3, const double *p2, int n, const double P[12])
{
    if (!p3 || !p2 || !P || n <= 0) return -1.0;
    double acc = 0.0;
    int used = 0;

    for (int i = 0; i < n; i++) {
        double X = p3[3 * i], Y = p3[3 * i + 1], Z = p3[3 * i + 2];
        double w = P[8] * X + P[9] * Y + P[10] * Z + P[11];
        if (fabs(w) < 1e-12) continue;
        double u = (P[0] * X + P[1] * Y + P[2] * Z + P[3]) / w;
        double v = (P[4] * X + P[5] * Y + P[6] * Z + P[7]) / w;
        double du = u - p2[2 * i], dv = v - p2[2 * i + 1];
        acc += du * du + dv * dv;
        used++;
    }
    if (!used) return -1.0;
    return sqrt(acc / used);
}


int wubu_pnp_decompose(const double P[12], double K[9], double R[9], double t[3])
{
    if (!P || !K || !R || !t) return -1;

    /* RQ by Gram-Schmidt from the bottom row up: row i of M is expressed in
     * the basis Q[i..2], so the coefficient matrix is upper triangular. */
    double M[3][3], Q[3][3], Rk[3][3];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) M[r][c] = P[4 * r + c];

    for (int i = 2; i >= 0; i--) {
        double v[3] = {M[i][0], M[i][1], M[i][2]};

        for (int j = i + 1; j < 3; j++) {
            double dot = 0.0;
            for (int c = 0; c < 3; c++) dot += v[c] * Q[j][c];
            Rk[i][j] = dot;
            for (int c = 0; c < 3; c++) v[c] -= dot * Q[j][c];
        }
        double norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (norm < 1e-12) return -1;
        Rk[i][i] = norm;
        for (int c = 0; c < 3; c++) Q[i][c] = v[c] / norm;
    }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < i; j++) Rk[i][j] = 0.0;

    /* Positive diagonal: flip row i of both factors together. */
    for (int i = 0; i < 3; i++) {
        if (Rk[i][i] < 0.0) {
            for (int c = 0; c < 3; c++) { Rk[i][c] = -Rk[i][c]; Q[i][c] = -Q[i][c]; }
        }
    }
    if (fabs(Rk[2][2]) < 1e-12) return -1;

    double scale = Rk[2][2];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) K[r * 3 + c] = Rk[r][c] / scale;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) R[r * 3 + c] = Q[r][c];

    double b[3] = {P[3] / scale, P[7] / scale, P[11] / scale};
    for (int i = 2; i >= 0; i--) {
        double acc = b[i];
        for (int j = i + 1; j < 3; j++) acc -= K[i * 3 + j] * t[j];
        if (fabs(K[i * 3 + i]) < 1e-12) return -1;
        t[i] = acc / K[i * 3 + i];
    }
    return 0;
}


static int solve8(double A[8][8], double b[8], double x[8])
{
    for (int col = 0; col < 8; col++) {
        int piv = col;
        for (int r = col + 1; r < 8; r++)
            if (fabs(A[r][col]) > fabs(A[piv][col])) piv = r;
        if (fabs(A[piv][col]) < 1e-12) return -1;
        if (piv != col) {
            for (int k = 0; k < 8; k++) { double t = A[col][k]; A[col][k] = A[piv][k]; A[piv][k] = t; }
            double t = b[col]; b[col] = b[piv]; b[piv] = t;
        }
        for (int r = col + 1; r < 8; r++) {
            double f = A[r][col] / A[col][col];
            if (f == 0.0) continue;
            for (int k = col; k < 8; k++) A[r][k] -= f * A[col][k];
            b[r] -= f * b[col];
        }
    }
    for (int i = 7; i >= 0; i--) {
        double s = b[i];
        for (int k = i + 1; k < 8; k++) s -= A[i][k] * x[k];
        if (fabs(A[i][i]) < 1e-12) return -1;
        x[i] = s / A[i][i];
    }
    return 0;
}

int wubu_homography(const double src[8], const double dst[8], double H[9])
{
    if (!src || !dst || !H) return -1;

    double A[8][8], b[8], h[8];
    for (int i = 0; i < 4; i++) {
        double x = src[2 * i], y = src[2 * i + 1];
        double u = dst[2 * i], v = dst[2 * i + 1];

        A[2 * i][0] = x; A[2 * i][1] = y; A[2 * i][2] = 1;
        A[2 * i][3] = 0; A[2 * i][4] = 0; A[2 * i][5] = 0;
        A[2 * i][6] = -u * x; A[2 * i][7] = -u * y;
        b[2 * i] = u;

        A[2 * i + 1][0] = 0; A[2 * i + 1][1] = 0; A[2 * i + 1][2] = 0;
        A[2 * i + 1][3] = x; A[2 * i + 1][4] = y; A[2 * i + 1][5] = 1;
        A[2 * i + 1][6] = -v * x; A[2 * i + 1][7] = -v * y;
        b[2 * i + 1] = v;
    }
    if (solve8(A, b, h) != 0) return -1;

    for (int i = 0; i < 8; i++) H[i] = h[i];
    H[8] = 1.0;
    return 0;
}

int wubu_warp_perspective(const wubu_img *src, const double H[9],
                          wubu_img *out, int ow, int oh)
{
    if (!src || !src->data || !H || !out || ow <= 0 || oh <= 0) return -1;
    if (wubu_img_alloc(out, ow, oh) != 0) return -1;

    /* Invert H (adjugate / det) so we can map destination pixels to source. */
    double a = H[0], b = H[1], c = H[2];
    double d = H[3], e = H[4], f = H[5];
    double g = H[6], h = H[7], i = H[8];
    double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (fabs(det) < 1e-12) return -1;

    double inv[9] = {
        (e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
        (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
        (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det
    };

    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
            double w = inv[6] * x + inv[7] * y + inv[8];
            if (fabs(w) < 1e-12) continue;
            double sx = (inv[0] * x + inv[1] * y + inv[2]) / w;
            double sy = (inv[3] * x + inv[4] * y + inv[5]) / w;

            int x0 = (int)floor(sx), y0 = (int)floor(sy);
            double fx = sx - x0, fy = sy - y0;
            double p00 = wubu_img_at(src, x0, y0), p10 = wubu_img_at(src, x0 + 1, y0);
            double p01 = wubu_img_at(src, x0, y0 + 1), p11 = wubu_img_at(src, x0 + 1, y0 + 1);
            double v = (p00 * (1 - fx) + p10 * fx) * (1 - fy)
                     + (p01 * (1 - fx) + p11 * fx) * fy;

            if (v < 0) v = 0;
            if (v > 255) v = 255;
            out->data[(size_t)y * out->stride + x] = (uint8_t)(v + 0.5);
        }
    }
    return 0;
}


int wubu_aruco_decode(const wubu_img *im, const double quad[8],
                      const uint16_t *dict, int ndict, int max_hamming,
                      wubu_aruco_marker *out)
{
    if (!im || !quad || !dict || !out || ndict <= 0 || max_hamming < 0) return -1;

    const int cells = 6;          /* 4x4 payload + 1-cell border */
    const int cell_px = 12;
    const int side = cells * cell_px;

    double dst[8] = {0, 0, side - 1, 0, side - 1, side - 1, 0, side - 1};
    double H[9];
    if (wubu_homography(quad, dst, H) != 0) return -1;

    wubu_img flat = {0};
    if (wubu_warp_perspective(im, H, &flat, side, side) != 0) return -1;

    int bits = 0;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            int px = (c + 2) * cell_px - cell_px / 2;   /* skip the border cell */
            int py = (r + 2) * cell_px - cell_px / 2;
            int v = wubu_img_at(&flat, px, py);
            if (v > 128) bits |= 1 << (15 - (r * 4 + c));
        }
    }
    wubu_img_free(&flat);

    int best = 99, best_id = -1;
    for (int i = 0; i < ndict; i++) {
        unsigned x = (unsigned)(bits ^ dict[i]);
        int dist = 0;
        while (x) { dist += (int)(x & 1u); x >>= 1; }
        if (dist < best) { best = dist; best_id = i; }
    }

    out->id = best_id;
    out->hamming = best;
    out->bits = bits;
    return best <= max_hamming ? 0 : -1;
}


int wubu_marker_pose(const double corners[8], double marker_size,
                     const double K[9], double R[9], double t[3])
{
    if (!corners || !K || !R || !t || marker_size <= 0.0) return -1;

    double s = marker_size;
    double plane[8] = {0, 0, s, 0, s, s, 0, s};
    double H[9];
    if (wubu_homography(plane, corners, H) != 0) return -1;

    /* Kinv: 3x3 inverse of K (assumed upper triangular, invertible). */
    double a = K[0], b = K[1], c = K[2];
    double d = K[3], e = K[4], f = K[5];
    double g = K[6], h = K[7], i = K[8];
    double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (fabs(det) < 1e-12) return -1;
    double Ki[9] = {
        (e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
        (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
        (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det
    };

    /* A = Kinv * H, column-major access via A[row*3+col]. */
    double A[9];
    for (int r = 0; r < 3; r++)
        for (int col = 0; col < 3; col++) {
            double acc = 0.0;
            for (int k = 0; k < 3; k++) acc += Ki[r * 3 + k] * H[k * 3 + col];
            A[r * 3 + col] = acc;
        }

    double n1 = sqrt(A[0] * A[0] + A[3] * A[3] + A[6] * A[6]);
    double n2 = sqrt(A[1] * A[1] + A[4] * A[4] + A[7] * A[7]);
    if (n1 < 1e-12 || n2 < 1e-12) return -1;
    double lambda = 2.0 / (n1 + n2);

    double r1[3] = {A[0] * lambda, A[3] * lambda, A[6] * lambda};
    double r2[3] = {A[1] * lambda, A[4] * lambda, A[7] * lambda};
    t[0] = A[2] * lambda; t[1] = A[5] * lambda; t[2] = A[8] * lambda;

    if (t[2] < 0.0) {
        for (int k = 0; k < 3; k++) { r1[k] = -r1[k]; r2[k] = -r2[k]; t[k] = -t[k]; }
    }

    /* Gram-Schmidt to kill noise, then r3 = r1 x r2. */
    double n = sqrt(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2]);
    if (n < 1e-12) return -1;
    for (int k = 0; k < 3; k++) r1[k] /= n;

    double dot = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2];
    for (int k = 0; k < 3; k++) r2[k] -= dot * r1[k];
    n = sqrt(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]);
    if (n < 1e-12) return -1;
    for (int k = 0; k < 3; k++) r2[k] /= n;

    double r3[3] = {
        r1[1] * r2[2] - r1[2] * r2[1],
        r1[2] * r2[0] - r1[0] * r2[2],
        r1[0] * r2[1] - r1[1] * r2[0]
    };

    for (int k = 0; k < 3; k++) {
        R[k * 3 + 0] = r1[k];
        R[k * 3 + 1] = r2[k];
        R[k * 3 + 2] = r3[k];
    }
    return 0;
}


static void rot_to_quat(const double R[9], double q[4])
{
    double tr = R[0] + R[4] + R[8];
    if (tr > 0.0) {
        double s = sqrt(tr + 1.0) * 2.0;
        q[0] = 0.25 * s;
        q[1] = (R[7] - R[5]) / s;
        q[2] = (R[2] - R[6]) / s;
        q[3] = (R[3] - R[1]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        double s = sqrt(1.0 + R[0] - R[4] - R[8]) * 2.0;
        q[0] = (R[7] - R[5]) / s;
        q[1] = 0.25 * s;
        q[2] = (R[1] + R[3]) / s;
        q[3] = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        double s = sqrt(1.0 + R[4] - R[0] - R[8]) * 2.0;
        q[0] = (R[2] - R[6]) / s;
        q[1] = (R[1] + R[3]) / s;
        q[2] = 0.25 * s;
        q[3] = (R[5] + R[7]) / s;
    } else {
        double s = sqrt(1.0 + R[8] - R[0] - R[4]) * 2.0;
        q[0] = (R[3] - R[1]) / s;
        q[1] = (R[2] + R[6]) / s;
        q[2] = (R[5] + R[7]) / s;
        q[3] = 0.25 * s;
    }
}

static void quat_to_rot(const double q[4], double R[9])
{
    double w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = 1 - 2 * (y * y + z * z);
    R[1] = 2 * (x * y - w * z);
    R[2] = 2 * (x * z + w * y);
    R[3] = 2 * (x * y + w * z);
    R[4] = 1 - 2 * (x * x + z * z);
    R[5] = 2 * (y * z - w * x);
    R[6] = 2 * (x * z - w * y);
    R[7] = 2 * (y * z + w * x);
    R[8] = 1 - 2 * (x * x + y * y);
}

int wubu_pose_fuse(const wubu_pose *in, int n, double R[9], double t[3])
{
    if (!in || !R || !t || n <= 0) return -1;

    double acc[4] = {0, 0, 0, 0};
    double wt = 0.0, tw = 0.0;
    double q0[4] = {1, 0, 0, 0};
    int have_ref = 0;

    for (int i = 0; i < n; i++) {
        double w = in[i].weight > 0.0 ? in[i].weight : 1.0;
        double q[4];
        rot_to_quat(in[i].R, q);

        if (!have_ref) { for (int k = 0; k < 4; k++) q0[k] = q[k]; have_ref = 1; }

        double dot = q0[0] * q[0] + q0[1] * q[1] + q0[2] * q[2] + q0[3] * q[3];
        double sign = dot < 0.0 ? -1.0 : 1.0;    /* hemisphere alignment */

        for (int k = 0; k < 4; k++) acc[k] += sign * w * q[k];
        for (int k = 0; k < 3; k++) t[k] += w * in[i].t[k];
        wt += w;
    }

    double norm = sqrt(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2] + acc[3] * acc[3]);
    if (norm < 1e-12 || wt <= 0.0) return -1;
    for (int k = 0; k < 4; k++) acc[k] /= norm;
    for (int k = 0; k < 3; k++) t[k] /= wt;

    (void)tw;
    quat_to_rot(acc, R);
    return 0;
}


static void project_pose(const double X[3], const double K[9], const double R[9],
                         const double t[3], double uv[2])
{
    double c[3];
    for (int i = 0; i < 3; i++)
        c[i] = R[i * 3 + 0] * X[0] + R[i * 3 + 1] * X[1] + R[i * 3 + 2] * X[2] + t[i];
    double w = K[6] * c[0] + K[7] * c[1] + K[8] * c[2];
    if (fabs(w) < 1e-12) w = 1e-12;
    uv[0] = (K[0] * c[0] + K[1] * c[1] + K[2] * c[2]) / w;
    uv[1] = (K[3] * c[0] + K[4] * c[1] + K[5] * c[2]) / w;
}

static double pose_cost(const double *p3, const double *p2, int n,
                        const double K[9], const double R[9], const double t[3])
{
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        double uv[2];
        project_pose(&p3[3 * i], K, R, t, uv);
        double du = uv[0] - p2[2 * i], dv = uv[1] - p2[2 * i + 1];
        acc += du * du + dv * dv;
    }
    return acc;
}

static int solve6(double A[6][6], double b[6], double x[6])
{
    for (int col = 0; col < 6; col++) {
        int piv = col;
        for (int r = col + 1; r < 6; r++)
            if (fabs(A[r][col]) > fabs(A[piv][col])) piv = r;
        if (fabs(A[piv][col]) < 1e-14) return -1;
        if (piv != col) {
            for (int k = 0; k < 6; k++) { double s = A[col][k]; A[col][k] = A[piv][k]; A[piv][k] = s; }
            double s = b[col]; b[col] = b[piv]; b[piv] = s;
        }
        for (int r = col + 1; r < 6; r++) {
            double f = A[r][col] / A[col][col];
            for (int k = col; k < 6; k++) A[r][k] -= f * A[col][k];
            b[r] -= f * b[col];
        }
    }
    for (int i = 5; i >= 0; i--) {
        double s = b[i];
        for (int k = i + 1; k < 6; k++) s -= A[i][k] * x[k];
        x[i] = s / A[i][i];
    }
    return 0;
}


static void exp_so3(const double w[3], double R[9])
{
    double theta = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    double k[9] = {0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0};
    double c, sn;
    if (theta < 1e-12) { c = 1.0; sn = 0.0; theta = 1e-12; }
    else { c = cos(theta); sn = sin(theta); }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double kij = k[i * 3 + j] / theta;
            double k2 = 0.0;
            for (int m = 0; m < 3; m++) k2 += (k[i * 3 + m] / theta) * (k[m * 3 + j] / theta);
            R[i * 3 + j] = (i == j ? 1.0 : 0.0) + sn * kij + (1.0 - c) * k2;
        }
}

int wubu_pose_refine(const double *p3, const double *p2, int n, const double K[9],
                     double R[9], double t[3], int iters)
{
    if (!p3 || !p2 || !n || n < 3 || !K || !R || !t || iters <= 0) return -1;

    double cur = pose_cost(p3, p2, n, K, R, t);

    for (int it = 0; it < iters; it++) {
        double A[6][6], b[6];
        memset(A, 0, sizeof A);
        memset(b, 0, sizeof b);

        for (int i = 0; i < n; i++) {
            double uv0[2];
            project_pose(&p3[3 * i], K, R, t, uv0);
            double r[2] = {p2[2 * i] - uv0[0], p2[2 * i + 1] - uv0[1]};
            if (sqrt(r[0] * r[0] + r[1] * r[1]) < 1e-14) continue;

            /* numeric Jacobian: 3 rotation params, 3 translation params */
            double J[2][6];
            const double eps = 1e-7;
            for (int k = 0; k < 6; k++) {
                double Rp[9], tp[3];
                memcpy(Rp, R, sizeof Rp);
                tp[0] = t[0]; tp[1] = t[1]; tp[2] = t[2];

                if (k < 3) {
                    double d[3] = {0, 0, 0};
                    d[k] = eps;
                    double dR[9];
                    exp_so3(d, dR);
                    double tmp[9];
                    for (int r0 = 0; r0 < 3; r0++)
                        for (int c0 = 0; c0 < 3; c0++) {
                            double s = 0;
                            for (int m = 0; m < 3; m++) s += dR[r0 * 3 + m] * R[m * 3 + c0];
                            tmp[r0 * 3 + c0] = s;
                        }
                    memcpy(Rp, tmp, sizeof Rp);
                } else {
                    tp[k - 3] += eps;
                }

                double uvp[2];
                project_pose(&p3[3 * i], K, Rp, tp, uvp);
                J[0][k] = (uvp[0] - uv0[0]) / eps;
                J[1][k] = (uvp[1] - uv0[1]) / eps;
            }

            for (int a = 0; a < 6; a++) {
                for (int c = 0; c < 6; c++) A[a][c] += J[0][a] * J[0][c] + J[1][a] * J[1][c];
                b[a] += J[0][a] * r[0] + J[1][a] * r[1];
            }
        }

        double dx[6];
        if (solve6(A, b, dx) != 0) return -1;

        /* Apply: R <- exp([dw]x) R, t += dt. */
        double Rold[9], told[3];
        memcpy(Rold, R, sizeof Rold);
        memcpy(told, t, sizeof told);

        double dw[3] = {dx[0], dx[1], dx[2]};
        double dR[9];
        exp_so3(dw, dR);
        double Rn[9];
        for (int r0 = 0; r0 < 3; r0++)
            for (int c0 = 0; c0 < 3; c0++) {
                double s = 0;
                for (int m = 0; m < 3; m++) s += dR[r0 * 3 + m] * R[m * 3 + c0];
                Rn[r0 * 3 + c0] = s;
            }
        memcpy(R, Rn, sizeof Rn);
        t[0] += dx[3]; t[1] += dx[4]; t[2] += dx[5];

        double nc = pose_cost(p3, p2, n, K, R, t);
        if (nc <= cur) {
            cur = nc;
        } else {
            memcpy(R, Rold, sizeof Rold);
            memcpy(t, told, sizeof told);
            break;               /* step made it worse; stop */
        }
        if (cur < 1e-16) break;
    }
    return 0;
}


int wubu_markers_refine(const double *marker_xyz, const double *marker_uv,
                        int nmarkers, const double K[9],
                        double R[9], double t[3], int iters)
{
    if (!marker_xyz || !marker_uv || nmarkers <= 0) return -1;
    int n = nmarkers * 4;
    return wubu_pose_refine(marker_xyz, marker_uv, n, K, R, t, iters);
}


static double focal_cost(const double *p3, const double *p2, int n,
                         double f0, double cx, double cy,
                         double R[9], double t[3])
{
    double K[9] = {f0, 0, cx, 0, f0, cy, 0, 0, 1};
    double Rl[9], tl[3];
    memcpy(Rl, R, sizeof Rl);
    memcpy(tl, t, sizeof tl);
    if (wubu_pose_refine(p3, p2, n, K, Rl, tl, 15) != 0) return 1e300;
    double cost = pose_cost(p3, p2, n, K, Rl, tl);
    return cost;
}

int wubu_calibrate_focal(const double *p3, const double *p2, int n,
                         double K[9], double R[9], double t[3])
{
    if (!p3 || !p2 || n < 4 || !K || !R || !t || K[0] <= 0) return -1;
    double cx = K[2], cy = K[5];
    double f_init = K[0];

    double lo = 0.3 * f_init, hi = 3.0 * f_init;
    const int samples = 60;
    double best_f = f_init, best_c = 1e300;
    double step = (hi - lo) / (samples - 1);

    for (int i = 0; i < samples; i++) {
        double f = lo + step * i;
        double c = focal_cost(p3, p2, n, f, cx, cy, R, t);
        if (c < best_c) { best_c = c; best_f = f; }
    }

    /* Golden-section around the best sample. */
    double a = best_f - step, b = best_f + step;
    if (a < lo) a = lo;
    if (b > hi) b = hi;
    const double gr = 0.6180339887498949;
    double x1 = b - gr * (b - a), x2 = a + gr * (b - a);
    double f1 = focal_cost(p3, p2, n, x1, cx, cy, R, t);
    double f2 = focal_cost(p3, p2, n, x2, cx, cy, R, t);

    for (int i = 0; i < 40; i++) {
        if (f1 < f2) { b = x2; x2 = x1; f2 = f1; x1 = b - gr * (b - a); f1 = focal_cost(p3, p2, n, x1, cx, cy, R, t); }
        else         { a = x1; x1 = x2; f1 = f2; x2 = a + gr * (b - a); f2 = focal_cost(p3, p2, n, x2, cx, cy, R, t); }
    }
    double f = (a + b) * 0.5;

    K[0] = f; K[4] = f;
    if (wubu_pose_refine(p3, p2, n, K, R, t, 40) != 0) return -1;
    return 0;
}


static double pp_cost(const double *p3, const double *p2, int n,
                      double f, double cx, double cy, const double R[9], const double t[3])
{
    double K[9] = {f, 0, cx, 0, f, cy, 0, 0, 1};
    double Rl[9], tl[3];
    memcpy(Rl, R, sizeof Rl);
    memcpy(tl, t, sizeof tl);
    if (wubu_pose_refine(p3, p2, n, K, Rl, tl, 15) != 0) return 1e300;
    return pose_cost(p3, p2, n, K, Rl, tl);
}

static double golden1d(double a, double b, int iters,
                       double (*fn)(const void *, double), const void *ctx)
{
    const double gr = 0.6180339887498949;
    double x1 = b - gr * (b - a), x2 = a + gr * (b - a);
    double f1 = fn(ctx, x1), f2 = fn(ctx, x2);
    for (int i = 0; i < iters; i++) {
        if (f1 < f2) { b = x2; x2 = x1; f2 = f1; x1 = b - gr * (b - a); f1 = fn(ctx, x1); }
        else         { a = x1; x1 = x2; f1 = f2; x2 = a + gr * (b - a); f2 = fn(ctx, x2); }
    }
    return (a + b) * 0.5;
}

typedef struct {
    const double *p3; const double *p2; int n;
    double f, other; int axis;          /* axis 0: vary cx, axis 1: vary cy */
    double R[9], t[3];
} pp_ctx;

static double pp_axis(const void *vctx, double val)
{
    const pp_ctx *c = (const pp_ctx *)vctx;
    double cx = c->axis == 0 ? val : c->other;
    double cy = c->axis == 0 ? c->other : val;
    return pp_cost(c->p3, c->p2, c->n, c->f, cx, cy, c->R, c->t);
}

int wubu_calibrate_principal(const double *p3, const double *p2, int n,
                             double K[9], double R[9], double t[3])
{
    if (!p3 || !p2 || n < 4 || !K || !R || !t || K[0] <= 0) return -1;

    double f = K[0], cx = K[2], cy = K[5];
    double span = 0.1 * f;

    for (int pass = 0; pass < 2; pass++) {
        pp_ctx c;
        c.p3 = p3; c.p2 = p2; c.n = n; c.f = f;
        memcpy(c.R, R, sizeof c.R);
        memcpy(c.t, t, sizeof c.t);

        c.axis = 0; c.other = cy;
        cx = golden1d(cx - span, cx + span, 30, pp_axis, &c);
        c.axis = 1; c.other = cx;
        cy = golden1d(cy - span, cy + span, 30, pp_axis, &c);
        span *= 0.25;                     /* narrow after the first pass */
    }

    K[2] = cx; K[5] = cy;
    if (wubu_pose_refine(p3, p2, n, K, R, t, 40) != 0) return -1;
    return 0;
}
