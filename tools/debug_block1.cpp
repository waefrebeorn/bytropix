/**
 * debug_block1.c — Debug why ggml produces zero for block 1.
 * Reads raw bytes for both block 0 and block 1, dequants with ggml.
 */
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s model.gguf file_offset raw_size type\n", argv[0]);
        return 1;
    }

    long file_off = atol(argv[2]);
    long raw_sz = atol(argv[3]);
    int type = atoi(argv[4]);

    FILE *f = fopen(argv[1], "rb"); fseek(f, file_off, SEEK_SET);
    uint8_t *raw = (uint8_t*)malloc(raw_sz);
    size_t nr = fread(raw, 1, raw_sz, f); fclose(f);
    if (nr != (size_t)raw_sz) { fprintf(stderr, "Read error\n"); free(raw); return 1; }

    int blk_sz = 256;
    int type_sz = ggml_type_size((enum ggml_type)type);
    int n_blocks = raw_sz / type_sz;
    int64_t n_elems = (int64_t)n_blocks * blk_sz;
    
    float *result = (float*)malloc(n_elems * sizeof(float));
    ggml_get_type_traits((enum ggml_type)type)->to_float(raw, result, n_blocks);

    // For block 0: check d, dmin as f16
    for (int b = 0; b < n_blocks && b < 2; b++) {
        const uint8_t *blk = raw + b * type_sz;
        
        // Read d and dmin as f16 (little-endian)
        uint16_t d_bits = blk[0] | (blk[1] << 8);
        uint16_t dmin_bits = blk[2] | (blk[3] << 8);
        
        // Decode using ggml's own method
        float d_ggml = ggml_fp16_to_fp32(d_bits);
        float dmin_ggml = ggml_fp16_to_fp32(dmin_bits);
        
        fprintf(stderr, "\nBlock %d:\n", b);
        fprintf(stderr, "  d raw=0x%04x via ggml=% .10f\n", d_bits, d_ggml);
        fprintf(stderr, "  dmin raw=0x%04x via ggml=% .10f\n", dmin_bits, dmin_ggml);
        
        // Check scales
        const uint8_t *scales = blk + 4;
        fprintf(stderr, "  scales: ");
        for (int i = 0; i < 12; i++) fprintf(stderr, "%02x ", scales[i]);
        fprintf(stderr, "\n");
        
        // Evaluate get_scale_min_k4 for is=0,1
        for (int is = 0; is < 8; is += 2) {
            uint8_t sc, m;
            if (is < 4) {
                sc = scales[is] & 63;
                m = scales[is+4] & 63;
            } else {
                sc = (scales[is+4] & 0xF) | ((scales[is-4] >> 6) << 4);
                m  = (scales[is+4] >>  4) | ((scales[is-0] >> 6) << 4);
            }
            fprintf(stderr, "  is=%d: sc=%d m=%d -> d1=% .10f m1=% .10f\n",
                    is, sc, m, d_ggml * sc, dmin_ggml * m);
        }
        
        // First 8 dequant values
        float *b_out = result + b * blk_sz;
        fprintf(stderr, "  First 8 dequantized: ");
        for (int i = 0; i < 8; i++) fprintf(stderr, "%+.10f ", b_out[i]);
        fprintf(stderr, "\n");
    }

    free(raw); free(result);
    return 0;
}
