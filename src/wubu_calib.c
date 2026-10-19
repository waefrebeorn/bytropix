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
