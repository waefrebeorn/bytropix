#include "gguf_reader.h"
#include "wubu_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>  // for isnan, isinf

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }
    
    const char *model_path = argv[1];
    int B = 1, T = 4;
    
    // Initialize model (auto-detects layers from tensor names)
    wubu_model_t model;
    if (!wubu_model_init(&model, model_path)) {
        fprintf(stderr, "Failed to init model\n"); return 1;
    }
    
    // Create random embedding
    float *embd = malloc(B * T * D_MODEL * sizeof(float));
    float *logits = malloc(B * T * D_MODEL * sizeof(float));
    for (int i = 0; i < B * T * D_MODEL; i++) embd[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    
    // Forward pass
    clock_t t0 = clock();
    wubu_model_forward_from_embd(&model, embd, B, T, logits);
    double t = (double)(clock() - t0) / CLOCKS_PER_SEC;
    
    // Stats
    int N = B * T;
    double sum = 0, sum2 = 0;
    float vmin = logits[0], vmax = logits[0];
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < N * D_MODEL; i++) {
        float v = logits[i];
        if (isnan(v)) { nan_count++; continue; }
        if (isinf(v)) { inf_count++; continue; }
        sum += v; sum2 += v*v;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
    int valid = N * D_MODEL - nan_count - inf_count;
    printf("Forward pass: %.3f s\n", t);
    printf("Hidden stats (%d valid / %d NaN / %d Inf):\n", valid, nan_count, inf_count);
    printf("  Range: [%.4f, %.4f]\n", vmin, vmax);
    printf("  Mean: %.6f, StdDev: %.6f\n", sum/valid, sqrt(sum2/valid - (sum/valid)*(sum/valid)));
    
    // Check for outliers
    int extreme = 0;
    for (int i = 0; i < N * D_MODEL; i++) {
        if (fabs(logits[i]) > 100.0f) extreme++;
    }
    printf("  |v|>100: %d / %d\n", extreme, N * D_MODEL);
    
    printf("\nVERDICT: ");
    if (nan_count > 0 || inf_count > 0) printf("FAIL (NaN/Inf in hidden states)\n");
    else if (extreme > 0) printf("WARN (extreme values in hidden states)\n");
    else printf("PASS (hidden states look reasonable)\n");
    
    free(embd); free(logits);
    wubu_model_free(&model);
    return 0;
}
