/**
 * Test dequant of full expert 64 gate weights.
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
    FILE *f = fopen("/tmp/dbg_expert_gate_raw.bin", "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *raw = (uint8_t *)malloc(sz);
    fread(raw, 1, sz, f); fclose(f);
    
    printf("Read %ld bytes (= %ld blocks)\n", sz, sz/66);
    
    int n_elems = sz / 66 * 256;
    float *out1 = (float *)malloc(n_elems * sizeof(float));
    float *out2 = (float *)malloc(n_elems * sizeof(float));
    
    dequantize_iq2_xxs_row(raw, out1, n_elems);
    dequantize_row_iq2_xxs((const block_iq2_xxs *)raw, out2, n_elems);
    
    double max_diff = 0;
    int n_bad = 0;
    for (int i = 0; i < n_elems; i++) {
        double d = fabs(out1[i] - out2[i]);
        if (d > max_diff) max_diff = d;
        if (d > 1e-5) n_bad++;
    }
    printf("First 8 byt: ");
    for (int i = 0; i < 8; i++) printf("%.10f ", out1[i]);
    printf("\nFirst 8 llm: ");
    for (int i = 0; i < 8; i++) printf("%.10f ", out2[i]);
    printf("\nmax_diff: %.10f, mismatches: %d/%d\n", max_diff, n_bad, n_elems);
    
    free(raw); free(out1); free(out2);
    return 0;
}
