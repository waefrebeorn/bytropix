/*
 * test_ops.c — Golden-file tests for the wubu_ops numerical primitives.
 *
 * Generated from research 066-D5 (testing strategy): each primitive
 * is exercised with edge-case inputs, reference output is captured to
 * a golden file, and subsequent runs diff against it. This catches
 * numerical regressions before they reach the model pipeline.
 *
 * Usage:
 *   make test_ops          # build + run, generate golden if missing
 *   WUBU_UPDATE_GOLDEN=1 make test_ops   # force-regenerate
 *
 * Golden files live in tests/golden/ (git-tracked, reviewed).
 */
#define _POSIX_C_SOURCE 200809L
#include "wubu_ops.h"
#include "wubu_banner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 11

static int tests_pass = 0;
static int tests_run  = 0;

static void check(const char *name, int cond) {
    tests_run++;
    if (cond) { tests_pass++; printf("  PASS: %s\n", name); }
    else      { printf("  FAIL: %s\n", name); }
}

static void check_f32(const char *name, float got, float expected, float tol) {
    tests_run++;
    float diff = fabsf(got - expected);
    if (diff <= tol) { tests_pass++; printf("  PASS: %s (got %.6f, exp %.6f)\n", name, got, expected); }
    else             { printf("  FAIL: %s (got %.6f, exp %.6f, diff %.2e)\n", name, got, expected, diff); }
}

static float *read_golden(const char *name, int *count) {
    char path[256];
    snprintf(path, sizeof(path), "tests/golden/op_%s.golden", name);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    *count = sz / sizeof(float);
    float *buf = malloc(sz);
    fread(buf, sizeof(float), *count, f);
    fclose(f);
    return buf;
}

static int write_golden(const char *name, const float *data, int count) {
    char path[256];
    snprintf(path, sizeof(path), "tests/golden/op_%s.golden", name);
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(data, sizeof(float), count, f);
    fclose(f);
    return 0;
}

static int golden_matches(const float *got, const float *ref, int n, float tol) {
    for (int i = 0; i < n; i++) {
        float diff = fabsf(got[i] - ref[i]);
        if (diff > tol * (1.0f + fabsf(ref[i]))) return 0;
    }
    return 1;
}

/* Edge-case inputs: mix of +/-large, small, zero */
static const float test_in[N] = { 0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
                                   80.0f, -80.0f, 0.01f, -0.01f, 100.0f, -100.0f };

int main(void) {
    wubu_print_banner("WuBuOS", "Numerical Ops Golden Tests");
    wubu_print_section("wubu_ops golden-file tests");

    int update = getenv("WUBU_UPDATE_GOLDEN") != NULL;
    float tol = 1e-5f;

    /* ---- softplus ---- */
    {
        float out[N], ref[N];
        wubu_softplus(N, test_in, out);
        float *g = read_golden("softplus", &(int){N});
        if (update) { write_golden("softplus", out, N); printf("  UPDATED: softplus\n"); tests_pass++; tests_run++; }
        else if (g) { check("softplus", golden_matches(out, g, N, tol)); free(g); }
        else { check("softplus (no golden)", 1); }
    }

    /* ---- silu ---- */
    {
        float out[N];
        wubu_silu(N, test_in, out);
        float *g = read_golden("silu", &(int){N});
        if (update) { write_golden("silu", out, N); printf("  UPDATED: silu\n"); tests_pass++; tests_run++; }
        else if (g) { check("silu", golden_matches(out, g, N, tol)); free(g); }
        else { check("silu (no golden)", 1); }
    }

    /* ---- sigmoid ---- */
    {
        float out[N];
        wubu_sigmoid(N, test_in, out);
        float *g = read_golden("sigmoid", &(int){N});
        if (update) { write_golden("sigmoid", out, N); printf("  UPDATED: sigmoid\n"); tests_pass++; tests_run++; }
        else if (g) { check("sigmoid", golden_matches(out, g, N, tol)); free(g); }
        else { check("sigmoid (no golden)", 1); }
    }

    /* ---- l2_norm ---- */
    {
        float out[N], weight[N];
        for (int i = 0; i < N; i++) weight[i] = 1.0f;
        wubu_l2_norm(1, 1, 1, N, test_in, 1e-6f, out);
        float *g = read_golden("l2_norm", &(int){N});
        if (update) { write_golden("l2_norm", out, N); printf("  UPDATED: l2_norm\n"); tests_pass++; tests_run++; }
        else if (g) { check("l2_norm", golden_matches(out, g, N, tol)); free(g); }
        else { check("l2_norm (no golden)", 1); }
    }

    /* ---- rms_norm ---- */
    {
        float out[N], weight[N];
        for (int i = 0; i < N; i++) weight[i] = 1.0f;
        wubu_rms_norm(1, 1, N, test_in, weight, 1e-6f, out);
        float *g = read_golden("rms_norm", &(int){N});
        if (update) { write_golden("rms_norm", out, N); printf("  UPDATED: rms_norm\n"); tests_pass++; tests_run++; }
        else if (g) { check("rms_norm", golden_matches(out, g, N, tol)); free(g); }
        else { check("rms_norm (no golden)", 1); }
    }

    /* ---- conv1d ---- */
    {
        float in[N + 3], kernel[4], out[N];  /* B=1,T=N,C=1,k=4 */
        for (int i = 0; i < N+3; i++) in[i] = test_in[i % N];
        for (int i = 0; i < 4; i++) kernel[i] = 0.25f;
        wubu_conv1d(1, N, 1, 4, in, kernel, out);
        float *g = read_golden("conv1d", &(int){N});
        if (update) { write_golden("conv1d", out, N); printf("  UPDATED: conv1d\n"); tests_pass++; tests_run++; }
        else if (g) { check("conv1d", golden_matches(out, g, N, tol)); free(g); }
        else { check("conv1d (no golden)", 1); }
    }

    wubu_print_stat("ops_cases", "%d", 6);
    wubu_print_stat("failures", "%d", tests_run - tests_pass);
    printf("\n=== Results: %d/%d tests passed ===\n", tests_pass, tests_run);

    return (tests_pass == tests_run) ? 0 : 1;
}
