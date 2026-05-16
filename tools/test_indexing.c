/**
 * Verify which weight indexing convention is correct.
 * Tests against known reference: for simple 2x2 matmul, verify y = x @ W 
 * where WGGUF is stored with dims=[input dim, output dim] = [2, 3]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    // Simulate GGUF storage: dims=[2, 3] means ne[0]=2, ne[1]=3
    // Stored as 3 rows x 2 columns, row-major
    // Conceptual W[2][3] stored as W_stored[3][2]
    // W[i][j] = W_stored[j][i] where W_stored at row j, col i
    // W_stored[row][col] = data[row * ne[0] + col] = data[row * 2 + col]
    
    // Let's create a known weight: W[i][j] = i*10 + j (conceptual)
    // So W = [[0, 1, 2], [10, 11, 12]]
    // W_stored = [[0, 10], [1, 11], [2, 12]]
    // Data: [0, 10, 1, 11, 2, 12]  (3 rows * 2 cols, row-major)
    
    float data[] = {0, 10, 1, 11, 2, 12}; // ne[0]=2, ne[1]=3
    const int DIM_IN = 2, DIM_OUT = 3;
    float x[] = {1, 2};  // input
    
    // Expected: y[j] = sum_i x[i] * W[i][j]
    // y[0] = 1*0 + 2*10 = 20
    // y[1] = 1*1 + 2*11 = 23
    // y[2] = 1*2 + 2*12 = 26
    
    float y_correct[] = {20, 23, 26};
    
    // Method A: weight[i * dim_out + j]  (the "wrong" one according to me)
    float yA[] = {0, 0, 0};
    for (int i = 0; i < DIM_IN; i++)
        for (int j = 0; j < DIM_OUT; j++)
            yA[j] += x[i] * data[i * DIM_OUT + j];
    
    // Method B: weight[i + j * dim_in]  (the "correct" one)
    float yB[] = {0, 0, 0};
    for (int i = 0; i < DIM_IN; i++)
        for (int j = 0; j < DIM_OUT; j++)
            yB[j] += x[i] * data[i + j * DIM_IN];
    
    printf("Expected: [%.0f, %.0f, %.0f]\n", y_correct[0], y_correct[1], y_correct[2]);
    printf("Method A (i*C+j): [%.0f, %.0f, %.0f]\n", yA[0], yA[1], yA[2]);
    printf("Method B (i+j*D): [%.0f, %.0f, %.0f]\n", yB[0], yB[1], yB[2]);
    
    // Now reverse: what if dims=[3, 2] (output dim first)?
    // ne=[3, 2], stored as 2 rows x 3 cols
    // W[2][3] conceptual, stored as W_stored[2][3]
    // Data: [0, 1, 2, 10, 11, 12]
    float data2[] = {0, 1, 2, 10, 11, 12};
    
    float yC[] = {0, 0, 0};
    for (int i = 0; i < DIM_IN; i++)
        for (int j = 0; j < DIM_OUT; j++)
            yC[j] += x[i] * data2[i * DIM_OUT + j];
    
    float yD[] = {0, 0, 0};
    for (int i = 0; i < DIM_IN; i++)
        for (int j = 0; j < DIM_OUT; j++)
            yD[j] += x[i] * data2[i + j * DIM_IN];
    
    printf("\nWith dims=[3,2] (output first):\n");
    printf("Expected: [%.0f, %.0f, %.0f]\n", y_correct[0], y_correct[1], y_correct[2]);
    printf("Method A (i*C+j): [%.0f, %.0f, %.0f]\n", yC[0], yC[1], yC[2]);
    printf("Method B (i+j*D): [%.0f, %.0f, %.0f]\n", yD[0], yD[1], yD[2]);
    
    return 0;
}
