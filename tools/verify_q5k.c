/**
 * verify_q5k.c — Verify Q5_K dequant against ggml reference.
 * 
 * Build:
 *   cd /home/wubu/bytropix
 *   g++ -std=c++11 -O2 -I include -I/home/wubu/llama.cpp/ggml/include \
 *       -o verify_q5k tools/verify_q5k.c src/gguf_reader.o \
 *       -L/home/wubu/llama.cpp/build/bin -Wl,-rpath,/home/wubu/llama.cpp/build/bin \
 *       -lggml-base -lggml-cpu -lm -lstdc++
 */
#include "gguf_reader.h"
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Our dequant function (declared in gguf_reader.c)
void dequantize_q5_K_row(const uint8_t *data, float *output, int64_t n_elems);

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s model.gguf tensor_name output_prefix\n", argv[0]);
        fprintf(stderr, "  Writes: {prefix}_our.bin and {prefix}_ref.bin\n");
        return 1;
    }

    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) return 1;

    gguf_tensor_info *target = NULL;
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, argv[2]) == 0) {
            target = &ctx->tensors[i];
            break;
        }
    }
    if (!target) { fprintf(stderr, "Tensor not found\n"); gguf_close(ctx); return 1; }

    int64_t n_elems = 1;
    for (int d = 0; d < target->n_dims; d++) n_elems *= target->dims[d];
    
    int64_t raw_size = gguf_raw_size(target->ggml_type, n_elems);
    uint64_t tensor_pos = ctx->data_blob_offset + target->data_offset;
    
    fprintf(stderr, "Tensor: %s type=%d n_elems=%ld raw_size=%ld\n",
            target->name, target->ggml_type, n_elems, raw_size);

    // Read raw data from file
    uint8_t *raw = (uint8_t *)malloc(raw_size);
    fseek(ctx->file, tensor_pos, SEEK_SET);
    size_t n_read = fread(raw, 1, raw_size, ctx->file);
    if (n_read != (size_t)raw_size) {
        fprintf(stderr, "Read error: %zu != %ld\n", n_read, raw_size);
        free(raw); gguf_close(ctx); return 1;
    }
    fprintf(stderr, "Read %zu raw bytes\n", n_read);

    // Dequant with OUR function
    float *our_out = (float *)malloc(n_elems * sizeof(float));
    dequantize_q5_K_row(raw, our_out, n_elems);

    // Dequant with ggml's reference
    float *ref_out = (float *)malloc(n_elems * sizeof(float));
    int64_t blck_size = 256;
    int64_t n_blocks = (n_elems + blck_size - 1) / blck_size;
    const struct ggml_type_traits *traits = ggml_get_type_traits(target->ggml_type);
    if (!traits || !traits->to_float) {
        fprintf(stderr, "ERROR: No dequant fn for type %d\n", target->ggml_type);
        free(raw); free(our_out); free(ref_out); gguf_close(ctx); return 1;
    }
    traits->to_float(raw, ref_out, n_blocks);
    fprintf(stderr, "Ref dequant: %ld blocks\n", n_blocks);

    // Compare block by block
    int blocks_matching = 0, blocks_mismatching = 0;
    int first_bad_block = -1;
    for (int64_t b = 0; b < n_blocks; b++) {
        int match = 1;
        for (int j = 0; j < blck_size; j++) {
            int64_t idx = b * blck_size + j;
            if (idx >= n_elems) break;
            float diff = fabsf(our_out[idx] - ref_out[idx]);
            if (diff > 1e-5f) { match = 0; break; }
        }
        if (match) blocks_matching++;
        else {
            blocks_mismatching++;
            if (first_bad_block < 0) first_bad_block = (int)b;
        }
    }
    fprintf(stderr, "\nBlock comparison: %d match, %d mismatch\n",
            blocks_matching, blocks_mismatching);
    if (first_bad_block >= 0) {
        int64_t b = first_bad_block;
        float omn=1e30f, omx=-1e30f, rmn=1e30f, rmx=-1e30f;
        for (int j = 0; j < blck_size && b*blck_size+j < n_elems; j++) {
            int64_t idx = b * blck_size + j;
            if (our_out[idx] < omn) omn = our_out[idx];
            if (our_out[idx] > omx) omx = our_out[idx];
            if (ref_out[idx] < rmn) rmn = ref_out[idx];
            if (ref_out[idx] > rmx) rmx = ref_out[idx];
        }
        fprintf(stderr, "  Block %d: our [%.4f, %.4f] ref [%.4f, %.4f]\n",
                (int)b, omn, omx, rmn, rmx);
        fprintf(stderr, "  First 8 our: ");
        for (int j = 0; j < 8 && b*blck_size+j < n_elems; j++)
            fprintf(stderr, "%+.6f ", our_out[b*blck_size + j]);
        fprintf(stderr, "\n  First 8 ref: ");
        for (int j = 0; j < 8 && b*blck_size+j < n_elems; j++)
            fprintf(stderr, "%+.6f ", ref_out[b*blck_size + j]);
        fprintf(stderr, "\n");
    }

    // Write output files
    char fn_our[512], fn_ref[512];
    snprintf(fn_our, sizeof(fn_our), "%s_our.bin", argv[3]);
    snprintf(fn_ref, sizeof(fn_ref), "%s_ref.bin", argv[3]);
    FILE *f = fopen(fn_our, "wb"); fwrite(our_out, 4, n_elems, f); fclose(f);
    f = fopen(fn_ref, "wb"); fwrite(ref_out, 4, n_elems, f); fclose(f);
    fprintf(stderr, "\nWrote %s and %s\n", fn_our, fn_ref);

    // Overall stats
    double o_sum=0, r_sum=0;
    float omn_all=1e30f, omx_all=-1e30f, rmn_all=1e30f, rmx_all=-1e30f;
    for (int64_t i = 0; i < n_elems; i++) {
        o_sum += our_out[i]; r_sum += ref_out[i];
        if (our_out[i] < omn_all) omn_all = our_out[i];
        if (our_out[i] > omx_all) omx_all = our_out[i];
        if (ref_out[i] < rmn_all) rmn_all = ref_out[i];
        if (ref_out[i] > rmx_all) rmx_all = ref_out[i];
    }
    fprintf(stderr, "\nOverall:\n");
    fprintf(stderr, "  Our:  mean=%.6f range=[%.4f, %.4f]\n",
            o_sum/n_elems, omn_all, omx_all);
    fprintf(stderr, "  Ref:  mean=%.6f range=[%.4f, %.4f]\n",
            r_sum/n_elems, rmn_all, rmx_all);
    
    // Cos sim
    double dot=0, onorm=0, rnorm=0;
    for (int64_t i = 0; i < n_elems; i++) {
        dot += our_out[i] * ref_out[i];
        onorm += our_out[i] * our_out[i];
        rnorm += ref_out[i] * ref_out[i];
    }
    fprintf(stderr, "  Cos sim: %.6f\n", dot / (sqrt(onorm) * sqrt(rnorm)));

    free(raw); free(our_out); free(ref_out);
    gguf_close(ctx);
    return 0;
}
