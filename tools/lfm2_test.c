/* lfm2_test.c -- verify LFM2.5 forward produces finite, sensible logits.
 * Loads the safetensors dir, embeds a toy token sequence, runs lfm2_forward,
 * prints logit stats + top-5 tokens. No BPE tokenizer needed (uses embed rows). */
#include "wubu_lfm2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <lfm2_dir> [tok0 tok1 ...]\n", argv[0]); return 1; }
    lfm2_model_t m;
    if (!lfm2_load(argv[1], &m)) { fprintf(stderr, "load failed\n"); return 1; }
    int T = argc - 2;
    if (T < 1) T = 1;
    float *emb = (float *)calloc((size_t)T * m.d_model, sizeof(float));
    for (int t = 0; t < T; t++) {
        int tok = (argc > 2 + t) ? atoi(argv[2 + t]) : 0;
        if (tok < 0 || tok >= m.vocab_size) tok = 0;
        const float *row = m.embed + (size_t)tok * m.d_model;
        memcpy(emb + (size_t)t * m.d_model, row, m.d_model * sizeof(float));
        printf("tok[%d]=%d\n", t, tok);
    }
    float *logits = (float *)calloc(m.vocab_size, sizeof(float));
    if (!lfm2_forward(&m, emb, 1, T, logits)) { fprintf(stderr, "forward failed\n"); return 1; }
    /* stats */
    float mx = -1e30f, mn = 1e30f, sum = 0.0f; int nan = 0, argmax = 0;
    for (int i = 0; i < m.vocab_size; i++) {
        float v = logits[i];
        if (isnan(v) || isinf(v)) { nan++; continue; }
        if (v > mx) { mx = v; argmax = i; }
        if (v < mn) mn = v;
        sum += v;
    }
    printf("[lfm2_test] vocab=%d finite=%d nan/inf=%d max=%.3f min=%.3f mean=%.4f argmax=%d\n",
           m.vocab_size, m.vocab_size - nan, nan, mx, mn, sum / m.vocab_size, argmax);
    /* top-10 (for comparison with HF reference) */
    for (int k = 0; k < 10; k++) {
        int bi = -1; float bv = -1e30f;
        for (int i = 0; i < m.vocab_size; i++) if (logits[i] > bv) { bv = logits[i]; bi = i; }
        printf("  top%d: tok=%d logit=%.4f\n", k, bi, bv);
        logits[bi] = -1e30f;
    }
    /* reference check indices (HF dumped these) */
    int checks[] = {0,100,1000,5000,99277,909,2028,14168,3431};
    printf("CHECK_LOGITS {");
    for (int c = 0; c < 9; c++) printf("\"%d\":%.4f%s", checks[c], logits[checks[c]], c<8?", ":"");
    printf("}\n");
    fflush(stdout);
    lfm2_free(&m);
    free(emb); free(logits);
    return nan ? 2 : 0;
}
