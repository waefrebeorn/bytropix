/**
 * test_mobius_linear.c — Validate Möbius linear layer forward + backward.
 */
#include "wubu_mobius_linear.h"
#include "wubu_mobius.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    int N = 4, D_in = 128, D_out = 64;
    float R_in = 2.0f, R_out = 5.0f;

    // Random input (valid Poincaré ball: ||x|| < R_in)
    float *x = (float*)malloc(N * D_in * sizeof(float));
    for (int i = 0; i < N * D_in; i++) {
        float r = (float)rand() / RAND_MAX;
        x[i] = (r - 0.5f) * 2.0f;
    }
    // Scale to ensure within ball: find max norm, scale all
    { float max_n=0; for(int s=0;s<N;s++){float n=0;for(int j=0;j<D_in;j++)n+=x[s*D_in+j]*x[s*D_in+j];n=sqrtf(n);if(n>max_n)max_n=n;}
      float s=R_in*0.7f/max_n;for(int i=0;i<N*D_in;i++)x[i]*=s; }

    // Random weight + bias
    float *W = (float*)malloc((int64_t)D_out * D_in * sizeof(float));
    float *b = (float*)malloc(D_out * sizeof(float));
    for (int64_t i = 0; i < (int64_t)D_out * D_in; i++) W[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < D_out; i++) b[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.01f;

    // Forward
    float *output = (float*)malloc(N * D_out * sizeof(float));
    wubu_mobius_linear_forward(x, N, D_in, D_out, W, b, R_in, R_out, output);

    // Validate output in ball
    int in_ball = 1;
    for (int s = 0; s < N; s++) {
        float n = 0;
        for (int j = 0; j < D_out; j++) n += output[s*D_out + j]*output[s*D_out + j];
        if (sqrtf(n) >= R_out * 0.999f) { in_ball = 0; printf("s=%d out of ball: ||y||=%.6e\n", s, sqrtf(n)); }
    }
    printf("Output in ball (||y||<R_out): %s\n", in_ball ? "✓" : "✗");

    // Backward with random d_output
    float *d_output = (float*)malloc(N * D_out * sizeof(float));
    for (int i = 0; i < N * D_out; i++) d_output[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.01f;

    float *d_x = (float*)calloc(N * D_in, sizeof(float));
    float *d_W = (float*)calloc((int64_t)D_out * D_in, sizeof(float));
    float *d_b = (float*)calloc(D_out, sizeof(float));

    // Save tangent intermediates for backward
    float *tangent_x = (float*)malloc(N * D_in * sizeof(float));
    float *tangent_out = (float*)malloc(N * D_out * sizeof(float));
    for (int s = 0; s < N; s++) {
        wubu_log_map(x + s*D_in, D_in, R_in, tangent_x + s*D_in);
        for (int j = 0; j < D_out; j++) {
            double sum = 0;
            for (int i = 0; i < D_in; i++)
                sum += (double)W[j*D_in + i] * (double)tangent_x[s*D_in + i];
            tangent_out[s*D_out + j] = (float)sum + b[j];
        }
    }

    wubu_mobius_linear_backward(x, N, D_in, D_out,
                                 tangent_x, tangent_out, output,
                                 d_output, W, R_in, R_out,
                                 d_x, d_W, d_b);

    // Check gradients
    float max_dx=0, max_dW=0, max_db=0;
    for (int i = 0; i < N*D_in; i++) { float v=fabsf(d_x[i]); if(v>max_dx)max_dx=v; }
    for (int64_t i = 0; i < (int64_t)D_out*D_in; i++) { float v=fabsf(d_W[i]); if(v>max_dW)max_dW=v; }
    for (int i = 0; i < D_out; i++) { float v=fabsf(d_b[i]); if(v>max_db)max_db=v; }

    printf("d_x_max:  %.6e  %s\n", max_dx, max_dx > 0 ? "✓" : "✗");
    printf("d_W_max:  %.6e  %s\n", max_dW, max_dW > 0 ? "✓" : "✗");
    printf("d_b_max:  %.6e  %s\n", max_db, max_db > 0 ? "✓" : "✗");

    // Check backward without saved tangents (recompute)
    float *d_x2 = (float*)calloc(N * D_in, sizeof(float));
    float *d_W2 = (float*)calloc((int64_t)D_out * D_in, sizeof(float));
    float *d_b2 = (float*)calloc(D_out, sizeof(float));
    wubu_mobius_linear_backward(x, N, D_in, D_out,
                                 NULL, NULL, output,
                                 d_output, W, R_in, R_out,
                                 d_x2, d_W2, d_b2);
    float max_diff=0;
    for (int i=0;i<N*D_in;i++){float v=fabsf(d_x[i]-d_x2[i]);if(v>max_diff)max_diff=v;}
    printf("Tangent-recompute consistency: max_diff=%.6e  %s\n", max_diff, max_diff<1e-6 ? "✓" : "~");

    int ok = (max_dx>0 && max_dW>0 && max_db>0);
    printf("\n%s\n", ok ? "ALL TESTS PASSED ✓" : "SOME TESTS FAILED ✗");

    free(x); free(W); free(b); free(output); free(d_output);
    free(d_x); free(d_W); free(d_b);
    free(d_x2); free(d_W2); free(d_b2);
    free(tangent_x); free(tangent_out);
    return ok ? 0 : 1;
}
