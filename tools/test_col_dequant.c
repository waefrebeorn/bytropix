/**
 * Quick test: dequantize IQ2_XXS column 0 from model data.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "ggml-common.h"
#include "ggml-quants.h"

void dequantize_iq2_xxs_row(const uint8_t *data, float *output, int64_t n_elems);

int main(void) {
    // Read raw column data (528 bytes = 8 blocks)
    FILE *f = fopen("/tmp/dbg_col0_raw.bin", "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *raw = (uint8_t *)malloc(sz);
    fread(raw, 1, sz, f); fclose(f);
    
    printf("Read %ld bytes\n", sz);
    
    // Dequantize with bytropix
    float out1[2048];
    dequantize_iq2_xxs_row(raw, out1, 2048);
    
    // Dequantize with llama.cpp  
    float out2[2048];
    dequantize_row_iq2_xxs((const block_iq2_xxs *)raw, out2, 2048);
    
    // Compare
    double max_diff = 0;
    for (int i = 0; i < 2048; i++) {
        double d = fabs(out1[i] - out2[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("bytropix[0..3]: %.10f %.10f %.10f %.10f\n", out1[0], out1[1], out1[2], out1[3]);
    printf("llamcpp[0..3]: %.10f %.10f %.10f %.10f\n", out2[0], out2[1], out2[2], out2[3]);
    printf("max_diff: %.10f\n", max_diff);
    
    free(raw);
    return 0;
}
