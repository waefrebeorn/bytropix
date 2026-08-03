/* test_gpu_ns5.c -- the GPU NS5 vs the CPU reference on square, wide,
 * and tall matrices. Caught the square-case-luck trap (the wrong Gram
 * GEMM layout only converges for square inputs). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "gpu_barun.h"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* the CPU NS5 (copied from the bp for the comparison) */
static void cpu_ns5(float *X, int rows, int cols)
{
    int trows = rows, tcols = cols;
    if (rows > cols) { trows = cols; tcols = rows; }
    float *tmp = (float *)malloc((size_t)rows * cols * sizeof(float));
    float *A = (float *)malloc((size_t)trows * trows * sizeof(float));
    float *B = (float *)malloc((size_t)trows * trows * sizeof(float));
    float *M = X;
    if (rows > cols) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) tmp[j * rows + i] = X[i * cols + j];
        M = tmp;
    }
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    for (int it = 0; it < 5; it++) {
        double ss = 0; for (int i = 0; i < trows * tcols; i++) ss += (double)M[i] * M[i];
        float nrm = (float)sqrt(ss);
        if (nrm > 1e-12f) for (int i = 0; i < trows * tcols; i++) M[i] /= nrm;
        for (int i = 0; i < trows; i++)
            for (int j = 0; j < trows; j++) {
                float acc = 0;
                for (int k = 0; k < tcols; k++) acc += M[i * tcols + k] * M[j * tcols + k];
                A[i * trows + j] = acc;
            }
        for (int i = 0; i < trows; i++)
            for (int j = 0; j < trows; j++) {
                float acc = 0;
                for (int k = 0; k < trows; k++) acc += A[i * trows + k] * A[k * trows + j];
                B[i * trows + j] = b * A[i * trows + j] + c * acc;
            }
        for (int i = 0; i < trows; i++)
            for (int j = 0; j < tcols; j++) {
                float acc = a * M[i * tcols + j];
                for (int k = 0; k < trows; k++) acc += B[i * trows + k] * M[k * tcols + j];
                M[i * tcols + j] = acc;
            }
    }
    if (rows > cols)
        for (int i = 0; i < trows; i++)
            for (int j = 0; j < tcols; j++) X[j * trows + i] = M[i * tcols + j];
    free(tmp); free(A); free(B);
}

int main(void)
{
    int shapes[][2] = { {448, 448}, {448, 2456}, {1228, 448}, {448, 64} };
    if (!gpu_barun_init()) { printf("SKIP (no CUDA device)\n"); return 0; }
    printf("gpu ready: %d\n", gpu_barun_ready());
    srand(42);
    for (int s = 0; s < 4; s++) {
        int rows = shapes[s][0], cols = shapes[s][1];
        float *g = (float *)malloc((size_t)rows * cols * sizeof(float));
        float *c = (float *)malloc((size_t)rows * cols * sizeof(float));
        float *orig2 = malloc((size_t)rows * cols * sizeof(float));
        for (int i = 0; i < rows * cols; i++) g[i] = c[i] = orig2[i] = (float)((rand() % 2000) - 1000) / 100.0f;
        double t0 = now_s();
        int ok = gpu_barun_ns5(g, rows, cols);
        double t1 = now_s();
        cpu_ns5(c, rows, cols);
        double t2 = now_s();
        double maxd = 0, sumg = 0, sumc = 0;
        for (int i = 0; i < rows * cols; i++) {
            double d = fabs((double)g[i] - (double)c[i]);
            if (d > maxd) maxd = d;
            sumg += fabs(g[i]); sumc += fabs(c[i]);
        }
        int pass = ok == 1 && maxd < 0.05 * (sumg / (rows * cols)) * 100;
        printf("  %dx%d: std rc=%d gpu %.1fms cpu %.1fms  max|g-c|=%.3e %s\n",
               rows, cols, ok, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, maxd,
               pass ? "OK" : "FAIL");
        if (!pass) { free(orig2); free(g); free(c); return 1; }

        /* the Gram variant (the square-space iteration, Tri Dao 2026) */
        float *gr = malloc((size_t)rows * cols * sizeof(float));
        for (int i = 0; i < rows * cols; i++) gr[i] = orig2[i];
        double tg0 = now_s();
        int gok = gpu_barun_ns5_gram(gr, rows, cols);
        double tg1 = now_s();
        maxd = 0; sumg = 0;
        for (int i = 0; i < rows * cols; i++) {
            double d = fabs((double)gr[i] - (double)c[i]);
            if (d > maxd) maxd = d;
            sumg += fabs(gr[i]);
        }
        int gpass = gok == 1 && maxd < 0.05 * (sumg / (rows * cols)) * 100;
        printf("  %dx%d: gram rc=%d gpu %.1fms  max|gram-cpu|=%.3e %s\n",
               rows, cols, gok, (tg1 - tg0) * 1000.0, maxd,
               gpass ? "OK" : "FAIL");
        free(orig2); free(g); free(c); free(gr);
        if (!gpass) return 1;
    }
    printf("ALL GPU NS5 TESTS PASSED -- the standard + Gram orthogonalization match the CPU reference\n");
    return 0;
}
