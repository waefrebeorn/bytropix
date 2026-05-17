/**
 * dequant_compare.c — Compare bytropix dequant vs llama.cpp ggml dequant.
 * Reads raw quantized data from GGUF file at known offset.
 * No gguf_reader.h include — avoids enum conflicts.
 *
 * Build:
 *   g++ -std=c++11 -O2 -I/home/wubu/llama.cpp/ggml/include \
 *       -o dequant_compare tools/dequant_compare.c src/gguf_reader.o \
 *       -L/home/wubu/llama.cpp/build/bin -Wl,-rpath,/home/wubu/llama.cpp/build/bin \
 *       -lggml-base -lggml-cpu -lm -lstdc++
 *
 * Usage:
 *   ./dequant_compare model.gguf file_offset raw_size ggml_type [tag]
 *
 * Example (Q5_K = 13, Q4_K = 12):
 *   python3 tools/dump_tensor_offsets.py blk.0.attn_qkv
 *   ./dequant_compare /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf <offset> <size> 13 q5k_attn_qkv
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// llama.cpp ggml header (defines enum ggml_type with the SAME values as ours)
extern "C" {
#include "ggml.h"
}

// Our dequant function — forward declared, no header needed
extern "C" void gguf_dequantize(const uint8_t *data, int ggml_type, int64_t n_elems, float *output);

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s model.gguf file_offset raw_size ggml_type [tag]\n", argv[0]);
        return 1;
    }

    const char *fname = argv[1];
    long file_offset = atol(argv[2]);
    long raw_size = atol(argv[3]);
    int ggml_type_i = atoi(argv[4]);
    enum ggml_type qtype = (enum ggml_type)ggml_type_i;
    const char *tag = (argc > 5) ? argv[5] : "dequant";

    // Read raw data from file at exact offset
    FILE *f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "ERROR: Cannot open %s\n", fname); return 1; }
    fseek(f, file_offset, SEEK_SET);

    uint8_t *raw = (uint8_t *)malloc(raw_size);
    size_t nr = fread(raw, 1, raw_size, f);
    fclose(f);
    if (nr != (size_t)raw_size) {
        fprintf(stderr, "ERROR: Read %zu bytes, expected %ld\n", nr, raw_size);
        free(raw); return 1;
    }
    fprintf(stderr, "[%s] Read %zu bytes from offset %ld, type=%d\n", tag, nr, file_offset, ggml_type_i);

    // Get block info from ggml
    int blck_size = ggml_blck_size(qtype);
    int type_size = ggml_type_size(qtype);
    int n_blocks = raw_size / type_size;
    int64_t n_elems = (int64_t)n_blocks * blck_size;
    fprintf(stderr, "  blck_size=%d type_size=%d n_blocks=%d n_elems=%ld\n",
            blck_size, type_size, n_blocks, n_elems);

    if (raw_size % type_size != 0) {
        fprintf(stderr, "  WARNING: raw_size %ld not multiple of type_size %d\n", raw_size, type_size);
    }

    // Dequant with our function (gguf_dequantize takes int type, not enum)
    float *our_out = (float *)malloc(n_elems * sizeof(float));
    gguf_dequantize(raw, ggml_type_i, n_elems, our_out);

    // Dequant with ggml's reference
    float *ref_out = (float *)malloc(n_elems * sizeof(float));
    const struct ggml_type_traits *traits = ggml_get_type_traits(qtype);
    if (!traits || !traits->to_float) {
        fprintf(stderr, "ERROR: No ggml dequant fn for type %d\n", ggml_type_i);
        free(raw); free(our_out); free(ref_out); return 1;
    }
    traits->to_float(raw, ref_out, n_elems);

    // Compare element by element
    int64_t total_match = 0, total_mismatch = 0;
    double max_diff = 0;
    int64_t max_diff_idx = -1;
    for (int64_t i = 0; i < n_elems; i++) {
        float diff = fabsf(our_out[i] - ref_out[i]);
        if (diff > max_diff) { max_diff = diff; max_diff_idx = i; }
        if (diff > 1e-5f) total_mismatch++;
        else total_match++;
    }

    fprintf(stderr, "\n=== [%s] RESULTS ===\n", tag);
    fprintf(stderr, "Match: %ld / %ld (%.2f%%)\n", (long)total_match, (long)n_elems,
            100.0 * total_match / n_elems);
    fprintf(stderr, "Max diff: %.10f at index %ld\n", max_diff, (long)max_diff_idx);

    // Show first 5 mismatching blocks
    int bad_shown = 0;
    for (int b = 0; b < n_blocks && bad_shown < 5; b++) {
        int block_bad = 0;
        float bmn_o=1e30f, bmx_o=-1e30f, bmn_r=1e30f, bmx_r=-1e30f;
        for (int j = 0; j < blck_size; j++) {
            int64_t idx = (int64_t)b * blck_size + j;
            if (idx >= n_elems) break;
            float d = fabsf(our_out[idx] - ref_out[idx]);
            if (d > 1e-5f) block_bad = 1;
            if (our_out[idx] < bmn_o) bmn_o = our_out[idx];
            if (our_out[idx] > bmx_o) bmx_o = our_out[idx];
            if (ref_out[idx] < bmn_r) bmn_r = ref_out[idx];
            if (ref_out[idx] > bmx_r) bmx_r = ref_out[idx];
        }
        if (block_bad) {
            int64_t bi = (int64_t)b * blck_size;
            fprintf(stderr, "\n  Block %d (elem %ld):\n", b, (long)bi);
            fprintf(stderr, "    Our [%.4f, %.4f]  Ref [%.4f, %.4f]\n", bmn_o, bmx_o, bmn_r, bmx_r);
            fprintf(stderr, "    Our first 8: ");
            for (int j = 0; j < 8 && bi+j < n_elems; j++)
                fprintf(stderr, "%+.6f ", our_out[bi+j]);
            fprintf(stderr, "\n    Ref first 8: ");
            for (int j = 0; j < 8 && bi+j < n_elems; j++)
                fprintf(stderr, "%+.6f ", ref_out[bi+j]);
            fprintf(stderr, "\n");
            bad_shown++;
        }
    }

    // Overall stats + cos sim
    double o_sum=0, r_sum=0, dot=0, on2=0, rn2=0;
    float omn=1e30f, omx=-1e30f, rmn=1e30f, rmx=-1e30f;
    for (int64_t i = 0; i < n_elems; i++) {
        o_sum += our_out[i]; r_sum += ref_out[i];
        dot   += our_out[i] * ref_out[i];
        on2   += our_out[i] * our_out[i];
        rn2   += ref_out[i] * ref_out[i];
        if (our_out[i] < omn) omn = our_out[i];
        if (our_out[i] > omx) omx = our_out[i];
        if (ref_out[i] < rmn) rmn = ref_out[i];
        if (ref_out[i] > rmx) rmx = ref_out[i];
    }
    double cos_sim = dot / (sqrt(on2) * sqrt(rn2));
    fprintf(stderr, "\nOverall:\n");
    fprintf(stderr, "  Our:  mean=%.6f range=[%.4f, %.4f]\n", o_sum/n_elems, omn, omx);
    fprintf(stderr, "  Ref:  mean=%.6f range=[%.4f, %.4f]\n", r_sum/n_elems, rmn, rmx);
    fprintf(stderr, "  Cos sim: %.10f\n", cos_sim);

    // Write bin dumps
    char fn_our[512], fn_ref[512];
    snprintf(fn_our, sizeof(fn_our), "/tmp/%s_our.bin", tag);
    snprintf(fn_ref, sizeof(fn_ref), "/tmp/%s_ref.bin", tag);
    f = fopen(fn_our, "wb"); fwrite(our_out, 4, n_elems, f); fclose(f);
    f = fopen(fn_ref, "wb"); fwrite(ref_out, 4, n_elems, f); fclose(f);
    fprintf(stderr, "\nWrote %s and %s\n", fn_our, fn_ref);

    if (total_mismatch == 0) {
        fprintf(stderr, "\n*** [%s] PASS: exact match ***\n", tag);
    } else {
        fprintf(stderr, "\n*** [%s] FAIL: %ld elements differ ***\n", tag, (long)total_mismatch);
    }

    free(raw); free(our_out); free(ref_out);
    return total_mismatch > 0 ? 1 : 0;
}
