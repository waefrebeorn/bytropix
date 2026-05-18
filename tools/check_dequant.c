/**
 * check_dequant.c — Verify IQ2_XXS dequant by comparing against ggml.
 * Build: g++ -std=c++11 -O2 -I /home/wubu/llama.cpp/include \
 *   -I /home/wubu/llama.cpp/ggml/include -I include \
 *   -o check_dequant tools/check_dequant.c \
 *   src/gguf_reader.o \
 *   -L /home/wubu/llama.cpp/build/bin -lllama -lggml-base -lggml-cpu -lggml \
 *   -lm -lstdc++ -Wl,-rpath,/home/wubu/llama.cpp/build/bin
 *
 * Usage: ./check_dequant model.gguf tensor_name expert_id
 */
#include "gguf_reader.h"
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

int main(int argc, char **argv) {
    if (argc < 4) { return 1; }
    const char *model_path = argv[1];
    const char *tensor_name = argv[2];
    int expert_id = atoi(argv[3]);
    
    // Load with our reader
    gguf_ctx *ctx = gguf_open(model_path);
    if (!ctx) return 1;
    
    gguf_tensor_info *t = gguf_find_tensor(ctx, tensor_name);
    if (!t || t->n_dims < 3) { return 1; }
    
    int64_t ne_per_exp = 1;
    for (int d = 0; d < t->n_dims - 1; d++) ne_per_exp *= t->dims[d];
    
    int64_t n_elems = ne_per_exp * t->dims[t->n_dims-1];
    int64_t raw_total = gguf_raw_size(t->ggml_type, n_elems);
    int64_t raw_per_exp = gguf_raw_size(t->ggml_type, ne_per_exp);
    int type = t->ggml_type;
    
    printf("Tensor: %s\n", tensor_name);
    printf("  dims: %d [", t->n_dims);
    for (int d = 0; d < t->n_dims; d++) printf("%lld%s", (long long)t->dims[d], d+1<t->n_dims?",":"");
    printf("]\n");
    printf("  type=%d\n", type);
    printf("  n_elems=%lld, ne_per_exp=%lld\n", (long long)n_elems, (long long)ne_per_exp);
    printf("  raw_total=%lld, raw_per_exp=%lld\n", (long long)raw_total, (long long)raw_per_exp);
    
    // Read raw quantized data
    // Get the raw data offset
    uint64_t tensor_pos = ctx->data_blob_offset + t->data_offset;
    
    // Read the raw bytes for the entire tensor
    uint8_t *raw_data = (uint8_t *)malloc(raw_total);
    fseek(ctx->file, tensor_pos, SEEK_SET);
    fread(raw_data, 1, raw_total, ctx->file);
    
    // Dequantize expert with our function
    float *our_deq = (float *)malloc(ne_per_exp * sizeof(float));
    const uint8_t *exp_raw = raw_data + expert_id * raw_per_exp;
    int block_bytes = (int)gguf_raw_size(type, 256);
    for (int64_t b = 0; b < ne_per_exp; b += 256) {
        int64_t nb = (ne_per_exp - b) > 256 ? 256 : (ne_per_exp - b);
        gguf_dequantize(exp_raw + (b/256) * block_bytes, type, nb, our_deq + b);
    }
    
    // Dequantize using ggml (llama.cpp's dequant)
    float *ggml_deq = (float *)malloc(ne_per_exp * sizeof(float));
    // Find the ggml dequant function. Use ggml_backend_tensor_get? No, that needs a tensor.
    // Let's use ggml's type dequantization directly if available
    // ggml has type_traits which has to_float function
    
    // Actually, let's use the ggml_dequantize_row function
    // GGML_API void ggml_dequantize_row(enum ggml_type type, const void *x, float *y, int64_t n);
    ggml_dequantize_row((ggml_type)type, exp_raw, ggml_deq, ne_per_exp);
    
    // Compare
    double sum=0, sumsq=0;
    float min_diff = 1e30f, max_diff = -1e30f;
    int n_match = 0, n_mismatch = 0;
    for (int64_t i = 0; i < ne_per_exp && i < 100000; i++) {
        float diff = our_deq[i] - ggml_deq[i];
        sum += diff;
        sumsq += diff * diff;
        if (diff < min_diff) min_diff = diff;
        if (diff > max_diff) max_diff = diff;
        if (fabs(diff) < 1e-6) n_match++;
        else n_mismatch++;
    }
    int64_t check = ne_per_exp < 100000 ? ne_per_exp : 100000;
    printf("\nComparison (first %lld of %lld elements):\n", (long long)check, (long long)ne_per_exp);
    printf("  mean diff = %.10f\n", (float)(sum/check));
    printf("  rms diff = %.10f\n", (float)(sqrt(sumsq/check)));
    printf("  min diff = %.10f, max diff = %.10f\n", min_diff, max_diff);
    printf("  elements matching: %d/%lld\n", n_match, (long long)(n_match+n_mismatch));
    
    // Show first 10 values
    printf("\nFirst 10 values (ours vs ggml):\n");
    for (int i = 0; i < 10; i++)
        printf("  [%d] %.8f vs %.8f (diff=%.8f)\n", i, our_deq[i], ggml_deq[i], our_deq[i]-ggml_deq[i]);
    
    free(raw_data);
    free(our_deq);
    free(ggml_deq);
    gguf_close(ctx);
    return 0;
}
