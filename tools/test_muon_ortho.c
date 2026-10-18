/* test_muon_ortho.c — BRAIN #244, first leg.
 *
 * Measures the polar orthogonalization error ||X^T X - I||_F on a 64x64
 * deterministic matrix. Records the 5-iteration error (the Muon quintic's
 * advertised budget) and requires the converged error to beat 1e-3.
 *
 * It also pins the reason this entry point exists: ns5_inplace's per-step
 * Frobenius renormalization pins the spectral norm at 1/sqrt(n) and never
 * orthogonalizes, so the public primitive uses the quadratic NS map with a
 * single normalization instead.
 */
#include "wubu_muon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else printf("  [PASS] %s\n", m); } while (0)

static double orth_err(const float *x, int n)
{
    double e = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += (double)x[(size_t)k * n + i] * (double)x[(size_t)k * n + j];
            double t = (i == j) ? 1.0 : 0.0;
            e += (s - t) * (s - t);
        }
    }
    return sqrt(e);
}

static void fill(float *x, int n, unsigned seed)
{
    unsigned s = seed;
    for (int i = 0; i < n * n; i++) {
        s = s * 1103515245u + 12345u;
        x[i] = (float)(((s >> 9) & 0xffff) / 32768.0) - 1.0f;
    }
}

int main(void)
{
    const int n = 64;
    float *a = malloc((size_t)n * n * sizeof(float));
    float *b = malloc((size_t)n * n * sizeof(float));
    if (!a || !b) { printf("alloc failed\n"); return 1; }

    fill(a, n, 12345);
    memcpy(b, a, (size_t)n * n * sizeof(float));

    CHECK(wubu_muon_orthogonalize(a, n, n, 5) == 0, "5-iteration call succeeds");
    double e5 = orth_err(a, n);

    CHECK(wubu_muon_orthogonalize(b, n, n, 25) == 0, "25-iteration call succeeds");
    double e25 = orth_err(b, n);

    printf("  ||X^T X - I||_F: 5 iters = %.3e, 25 iters = %.3e\n", e5, e25);

    CHECK(e25 < 1e-3, "converged orthogonalization < 1e-3 (25 iterations)");
    CHECK(e25 < e5, "error decreases with more iterations");

    /* wide matrix: exercises the transpose path */
    int rows = 32, cols = 64;
    float *w = malloc((size_t)rows * cols * sizeof(float));
    unsigned s = 999;
    for (int i = 0; i < rows * cols; i++) {
        s = s * 1103515245u + 12345u;
        w[i] = (float)(((s >> 9) & 0xffff) / 32768.0) - 1.0f;
    }
    CHECK(wubu_muon_orthogonalize(w, rows, cols, 25) == 0, "wide matrix runs");
    /* X X^T should be I (32x32) */
    double e = 0.0;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < rows; j++) {
            double t = 0.0;
            for (int k = 0; k < cols; k++)
                t += (double)w[(size_t)i * cols + k] * (double)w[(size_t)j * cols + k];
            double d = t - (i == j ? 1.0 : 0.0);
            e += d * d;
        }
    printf("  wide 32x64 ||X X^T - I||_F = %.3e\n", sqrt(e));
    CHECK(sqrt(e) < 1e-3, "wide matrix orthogonalizes on the row space");

    CHECK(wubu_muon_orthogonalize(NULL, 4, 4, 5) == -1, "NULL -> -1");
    CHECK(wubu_muon_orthogonalize(w, 0, 4, 5) == -1, "zero rows -> -1");
    CHECK(wubu_muon_orthogonalize(w, 4, 4, 0) == -1, "zero iters -> -1");

    free(a); free(b); free(w);
    printf("\n=== test_muon_ortho: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}