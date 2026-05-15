#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    wubu_model_t model;
    if (!wubu_model_init(&model, path)) return 1;
    
    gguf_ctx *ctx = model.gguf_ctx;
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    printf("output.weight: dims=[%ld,%ld] type=%d\n", t->dims[0], t->dims[1], t->ggml_type);
    
    int n_vocab = 10;
    int n_elems = n_vocab * D_MODEL;
    float *w = (float *)malloc(n_elems * sizeof(float));
    gguf_read_tensor_f32(ctx, t, w, n_elems);
    
    printf("\n=== w[j*2048+0] for j=0..9 (first dim of each vocab): ===\n");
    for (int j = 0; j < n_vocab; j++) {
        printf("  out[%d][0]=%.6e\n", j, w[j*2048+0]);
    }
    
    // Sum of weight row for each vocab entry
    float sums[10];
    for (int j = 0; j < n_vocab; j++) {
        double s = 0;
        for (int k = 0; k < 2048; k++) s += w[j*2048 + k];
        sums[j] = (float)s;
    }
    float mean=0, var=0, minv=1e30, maxv=-1e30;
    for (int j = 0; j < n_vocab; j++) {
        mean += sums[j];
        if (sums[j] < minv) minv = sums[j];
        if (sums[j] > maxv) maxv = sums[j];
    }
    mean /= n_vocab;
    for (int j = 0; j < n_vocab; j++) var += (sums[j]-mean)*(sums[j]-mean);
    var /= n_vocab;
    printf("\nWeight sums across first %d vocab:\n", n_vocab);
    printf("  mean=%.4f std=%.4f range=[%.4f,%.4f]\n", mean, sqrtf(var), minv, maxv);
    
    free(w);
    wubu_model_free(&model);
    printf("=== PASS ===\n");
    return 0;
}
