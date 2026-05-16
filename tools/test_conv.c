/**
 * Verify conv1d weight indexing against known reference pattern.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    // Conv weight stored with dims=[4, 8192], ne[0]=4, ne[1]=8192
    // Row-major: 8192 rows x 4 cols
    // Weight[channel][tap] = data[tap + channel * 4]
    
    // Test: simulate conv weight with known values
    // Channel c, tap k: W[c][k] = c * 100 + k
    float *conv_w = (float *)malloc(4 * 8192 * sizeof(float));
    for (int c = 0; c < 8192; c++)
        for (int k = 0; k < 4; k++)
            conv_w[k + c * 4] = c * 100 + k;  // weight[tap + channel * k]
    
    // Verify our indexing
    printf("conv_w[channel=0][tap=0..3]: ");
    for (int k = 0; k < 4; k++) printf("%.0f ", conv_w[k + 0 * 4]);
    printf(" (expected: 0 1 2 3)\n");
    
    printf("conv_w[channel=1][tap=0..3]: ");
    for (int k = 0; k < 4; k++) printf("%.0f ", conv_w[k + 1 * 4]);
    printf(" (expected: 100 101 102 103)\n");
    
    printf("conv_w[channel=5][tap=0..3]: ");
    for (int k = 0; k < 4; k++) printf("%.0f ", conv_w[k + 5 * 4]);
    printf(" (expected: 500 501 502 503)\n");
    
    // Now test the convolution itself
    int B=1, T=2, C=8192, K=4;
    float input[(T+K-1) * C];
    memset(input, 0, sizeof(input));
    input[(K-1) * C + 0] = 1.0f;  // first channel, first real input
    input[(K-1) * C + 1] = 2.0f;  // second channel, first real input
    
    float output[T * C];
    memset(output, 0, sizeof(output));
    
    // Our conv1d implementation
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                float sum = 0.0f;
                for (int ki = 0; ki < K; ki++) {
                    int t_in = t + ki;
                    sum += input[(b * (T + K - 1) + t_in) * C + c] *
                           conv_w[ki + c * K];
                }
                output[(b * T + t) * C + c] = sum;
            }
        }
    }
    
    printf("\nConv output [t=0][c=0]: %.0f (expected: 0)\n", output[0]);   // relies on padding
    printf("Conv output [t=0][c=1]: %.0f (expected: 0)\n", output[1]);   // relies on padding
    printf("Conv output [t=1][c=0]: %.0f (expected: %d)\n", output[C+0], 0);  // input[K-1][0]=1.0 * weight[0][0]
    printf("Conv output [t=1][c=1]: %.0f (expected: %d)\n", output[C+1], 2);  // input[K-1][1]=2.0 * weight[1][1]
    
    // Check: for t=1, c=0: sum over k input[1+k][0] * conv_w[k][0]
    // input[1][0]=1.0, input[2][0]=0, input[3][0]=0, input[4][0]=0
    // conv_w[0][0]=0, conv_w[1][0]=1, conv_w[2][0]=2, conv_w[3][0]=3
    // sum = 1.0 * 0 + 0 + 0 + 0 = 0
    printf("  manual check: 1.0 * %.0f = %.0f\n", conv_w[0+0*4], 1.0f*conv_w[0+0*4]);
    
    free(conv_w);
    return 0;
}
