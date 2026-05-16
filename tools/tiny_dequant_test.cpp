/**
 * tiny_dequant_test.cpp — Minimal test: dequant a GGUF Q5_K tensor using ggml's dequant.
 * Reads raw data directly via fseek/fread, avoiding any gguf parser issues.
 *
 * Build:
 *   g++ -std=c++11 -O2 -I/home/wubu/llama.cpp/ggml/include \
 *       -o /tmp/tiny_dequant tiny_dequant_test.cpp \
 *       -L/home/wubu/llama.cpp/build/bin -Wl,-rpath,/home/wubu/llama.cpp/build/bin \
 *       -lggml-base -lggml-cpu -lm -lstdc++
 *
 * Usage: /tmp/tiny_dequant model.gguf tensor_offset raw_size type output.bin
 *   tensor_offset = byte offset from start of file to tensor data
 *   raw_size = number of bytes of quantized data
 *   type = ggml_type enum value (13=Q5_K)
 */
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s model.gguf file_offset raw_size ggml_type output.bin\n", argv[0]);
        return 1;
    }

    const char *fname = argv[1];
    long file_offset = atol(argv[2]);
    long raw_size = atol(argv[3]);
    int ggml_type = atoi(argv[4]);
    const char *outname = argv[5];

    // Read raw data from file at exact offset
    FILE *f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname); return 1; }
    fseek(f, file_offset, SEEK_SET);
    
    std::vector<uint8_t> raw(raw_size);
    size_t nr = fread(raw.data(), 1, raw_size, f);
    fclose(f);
    if (nr != (size_t)raw_size) {
        fprintf(stderr, "Read error: %zu != %ld\n", nr, raw_size);
        return 1;
    }
    fprintf(stderr, "Read %zu bytes from offset %ld\n", nr, file_offset);

    // Get block info
    int blck_size = ggml_blck_size((enum ggml_type)ggml_type);
    int n_blocks = raw_size / ggml_type_size((enum ggml_type)ggml_type);
    int64_t n_elems = (int64_t)n_blocks * blck_size;
    fprintf(stderr, "type=%d blck=%d n_blocks=%d n_elems=%ld\n",
            ggml_type, blck_size, n_blocks, n_elems);

    // Dequant
    std::vector<float> float_buf(n_elems);
    auto traits = ggml_get_type_traits((enum ggml_type)ggml_type);
    if (!traits || !traits->to_float) {
        fprintf(stderr, "No dequant fn for type %d\n", ggml_type);
        return 1;
    }
    traits->to_float(raw.data(), float_buf.data(), n_blocks);
    fprintf(stderr, "Dequantized %d blocks -> %ld floats\n", n_blocks, n_elems);

    // Write output
    FILE *fo = fopen(outname, "wb");
    if (fo) { fwrite(float_buf.data(), 4, n_elems, fo); fclose(fo); }
    fprintf(stderr, "Wrote %s\n", outname);

    // Stats per block
    for (int b = 0; b < n_blocks && b < 270; b++) {
        float bmn=1e30f, bmx=-1e30f;
        int nz = 0;
        for (int j = 0; j < blck_size && (int64_t)(b*blck_size+j) < n_elems; j++) {
            float v = float_buf[b*blck_size + j];
            if (v < bmn) bmn = v;
            if (v > bmx) bmx = v;
            if (fabsf(v) > 1e-10f) nz++;
        }
        if (b < 10 || (b >= 254 && b < 262))
            fprintf(stderr, "  Block %3d: range=[%.4f, %.4f] nonzero=%d/%d\n",
                    b, bmn, bmx, nz, blck_size);
    }

    // Overall stats
    float mn=float_buf[0], mx=float_buf[0];
    double sum=0;
    for (int64_t i = 0; i < n_elems; i++) {
        sum += float_buf[i];
        if (float_buf[i] < mn) mn = float_buf[i];
        if (float_buf[i] > mx) mx = float_buf[i];
    }
    fprintf(stderr, "\nOverall: mean=%.6f range=[%.4f, %.4f]\n",
            sum/n_elems, mn, mx);

    return 0;
}
