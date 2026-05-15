/**
 * test_text_fwd.c — test model forward with text embeddings
 */
#include "wubu_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    wubu_model_t model;
    if (!wubu_model_init(&model, "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    
    FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
    if (!f) { perror("fopen"); return 1; }
    float embd[8*2048];
    fread(embd, sizeof(float), 8*2048, f);
    fclose(f);
    
    double nrm = 0;
    for (int i = 0; i < 2048; i++) nrm += embd[i]*embd[i];
    printf("Text embd[0] norm=%.4f\n", sqrt(nrm));
    
    double t0 = now();
    float *logits = (float *)malloc(8 * model.vocab_size * sizeof(float));
    wubu_model_forward_from_embd(&model, embd, 1, 8, logits);
    printf("Forward: %.3f s\n", now() - t0);
    
    int nan_c = 0; float min_v=1e30, max_v=-1e30;
    for (int i = 0; i < 8 * model.vocab_size; i++) {
        if (isnan(logits[i])) nan_c++;
        if (logits[i] > max_v) max_v = logits[i];
        if (logits[i] < min_v) min_v = logits[i];
    }
    printf("Logits: nan=%d/%d range=[%.4f,%.4f]\n", nan_c, 8*model.vocab_size, min_v, max_v);
    printf("First[0:4]: %.4f %.4f %.4f %.4f\n", logits[0], logits[1], logits[2], logits[3]);
    
    wubu_model_free(&model);
    free(logits);
    return nan_c > 0;
}
