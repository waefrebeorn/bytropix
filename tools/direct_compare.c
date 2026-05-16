/**
 * direct_compare.c — Compare our Q5_K dequant vs ggml reference on SAME raw data.
 * Reads raw bytes directly from file at given offset, dequants both ways.
 *
 * Build:
 *   g++ -std=c++11 -O2 -I include -I/home/wubu/llama.cpp/ggml/include \
 *       -o /tmp/direct_compare tools/direct_compare.c src/gguf_reader.o \
 *       -L/home/wubu/llama.cpp/build/bin -Wl,-rpath,/home/wubu/llama.cpp/build/bin \
 *       -lggml-base -lggml-cpu -lm -lstdc++
 */
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Our dequant function (C linkage — compiled as C in gguf_reader.o)
extern "C" void dequantize_q5_K_row(const uint8_t *data, float *output, int64_t n_elems);

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s model.gguf file_offset raw_size output_prefix\n", argv[0]);
        return 1;
    }

    long file_off = atol(argv[2]);
    long raw_sz = atol(argv[3]);

    // Read raw data
    FILE *f = fopen(argv[1], "rb"); fseek(f, file_off, SEEK_SET);
    uint8_t *raw = (uint8_t*)malloc(raw_sz);
    size_t nr = fread(raw, 1, raw_sz, f); fclose(f);
    if (nr != (size_t)raw_sz) { fprintf(stderr, "Read error\n"); free(raw); return 1; }

    int blk_sz = 256;
    int n_blocks = raw_sz / 176;  // Q5_K type_size = 176
    int64_t n_elems = (int64_t)n_blocks * blk_sz;
    fprintf(stderr, "Read %zu bytes, %d blocks, %ld elems\n", nr, n_blocks, n_elems);

    // Our dequant
    float *our = (float*)malloc(n_elems * sizeof(float));
    dequantize_q5_K_row(raw, our, n_elems);

    // ggml dequant
    float *ref = (float*)malloc(n_elems * sizeof(float));
    ggml_get_type_traits(GGML_TYPE_Q5_K)->to_float(raw, ref, n_blocks);

    // Compare element-by-element and block-by-block
    int total_diff = 0;
    int first_diff_block = -1;
    for (int b = 0; b < n_blocks; b++) {
        int differs = 0;
        for (int j = 0; j < blk_sz; j++) {
            int64_t idx = b * blk_sz + j;
            if (idx >= n_elems) break;
            if (fabsf(our[idx] - ref[idx]) > 1e-5f) {
                differs = 1;
                total_diff++;
                if (first_diff_block < 0) first_diff_block = b;
            }
        }
        // Print first 14 blocks and blocks around boundary
        if (b < 14 || (b >= 254 && b < 262)) {
            float omx=-1e30f, omn=1e30f, rmx=-1e30f, rmn=1e30f;
            int onz=0, rnz=0;
            for (int j = 0; j < blk_sz; j++) {
                int64_t idx = b * blk_sz + j;
                if (idx >= n_elems) break;
                if (our[idx] < omn) omn=our[idx]; if (our[idx] > omx) omx=our[idx];
                if (ref[idx] < rmn) rmn=ref[idx]; if (ref[idx] > rmx) rmx=ref[idx];
                if (fabsf(our[idx]) > 1e-10f) onz++;
                if (fabsf(ref[idx]) > 1e-10f) rnz++;
            }
            const char *d = differs ? " <<< DIFF" : "";
            fprintf(stderr, "Block %3d: our[%.4f,%.4f nz=%d] ref[%.4f,%.4f nz=%d]%s\n",
                    b, omn, omx, onz, rmn, rmx, rnz, d);
        }
    }

    fprintf(stderr, "\nTotal differing elements: %d/%ld (%.1f%%)\n",
            total_diff, n_elems, 100.0 * total_diff / n_elems);
    if (first_diff_block >= 0)
        fprintf(stderr, "First differing block: %d\n", first_diff_block);

    // Write files
    char fn_our[256], fn_ref[256];
    snprintf(fn_our, sizeof(fn_our), "%s_our_q5k.bin", argv[4]);
    snprintf(fn_ref, sizeof(fn_ref), "%s_ref_q5k.bin", argv[4]);
    FILE *fo = fopen(fn_our, "wb"); fwrite(our, 4, n_elems, fo); fclose(fo);
    fo = fopen(fn_ref, "wb"); fwrite(ref, 4, n_elems, fo); fclose(fo);
    fprintf(stderr, "Wrote %s and %s\n", fn_our, fn_ref);

    free(raw); free(our); free(ref);
    return 0;
}
