/**
 * compare_weights.c — Dump quantized weight from both our reader and llama.cpp.
 * Build: 
 * gcc -O2 -I include -o compare_weights tools/compare_weights.c \
 *     src/gguf_reader.o src/wubu_ssm.o src/wubu_mobius.o \
 *     src/wubu_moe.o src/wubu_tokenizer.o src/qlearner.o \
 *     src/wubu_model.o src/wubu_ssm_chunked.o src/wubu_nested_ssm.o \
 *     src/wubu_nested_ssm_backward.o src/wubu_moe_backward.o \
 *     src/wubu_moe_hyperbolic.o src/wubu_poincare_ssm_backward.o \
 *     src/wubu_poincare_gqa.o src/wubu_poincare_gqa_backward.o \
 *     src/wubu_mobius_linear.o src/wubu_hyperbolic_output_proj.o \
 *     src/wubu_vision.o src/dequant_iq2_xxs.o src/rsgd.o src/wubu_tst.o \
 *     -lm -fopenmp -L/usr/local/cuda/lib64 -lcublas -lcudart -lstdc++
 *
 * Usage: ./compare_weights model.gguf tensor_name
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 3) { 
        fprintf(stderr, "Usage: %s model.gguf tensor_name\n", argv[0]); 
        return 1; 
    }
    
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "Failed open\n"); return 1; }
    
    const char *tensor_name = argv[2];
    gguf_tensor_info *t = gguf_find_tensor(ctx, tensor_name);
    if (!t) { fprintf(stderr, "Tensor '%s' not found\n", tensor_name); return 1; }
    
    int64_t n_elems = 1;
    for (int d = 0; d < t->n_dims; d++) n_elems *= t->dims[d];
    
    printf("Tensor '%s': dims=%d [", tensor_name, t->n_dims);
    for (int d = 0; d < t->n_dims; d++) {
        printf("%lld%s", (long long)t->dims[d], d+1<t->n_dims ? "," : "");
    }
    printf("] type=%d\n", t->ggml_type);
    fflush(stdout);
    
    // Read full tensor
    float *data = (float *)malloc(n_elems * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, data, n_elems)) {
        fprintf(stderr, "Failed to read tensor (n_elems=%lld)\n", (long long)n_elems);
        free(data);
        gguf_close(ctx);
        return 1;
    }
    
    printf("Read %lld elements\n", (long long)n_elems);
    
    // Stats
    double sum=0, sumsq=0;
    float minv = 1e30f, maxv = -1e30f;
    for (int64_t i = 0; i < n_elems && i < 10000000; i++) {
        sum += data[i];
        sumsq += data[i] * data[i];
        if (data[i] < minv) minv = data[i];
        if (data[i] > maxv) maxv = data[i];
    }
    printf("Stats (first 10M elems): mean=%.6f rms=%.6f min=%.6f max=%.6f\n",
           (float)(sum/n_elems), (float)sqrt(sumsq/n_elems), minv, maxv);
    
    // Dump to file
    char fn[256];
    snprintf(fn, sizeof(fn), "/tmp/our_%s.bin", tensor_name[0]=='_' ? tensor_name+5 : tensor_name);
    // Replace dots with underscores
    for (char *p = fn; *p; p++) if (*p == '.') *p = '_';
    FILE *f = fopen(fn, "wb");
    if (f) {
        // Only dump first 16384 elements to keep file small
        int64_t dump_n = n_elems < 16384 ? n_elems : 16384;
        fwrite(data, sizeof(float), dump_n, f);
        fclose(f);
    }
    printf("Saved to %s (first %lld elems)\n", fn, (long long)(n_elems<16384?n_elems:16384));
    
    free(data);
    gguf_close(ctx);
    return 0;
}
