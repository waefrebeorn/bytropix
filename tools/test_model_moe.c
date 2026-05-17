/**
 * test_model_moe.c — Run wubu_model_forward with MoE enabled, dump logits.
 * Build: gcc -O2 -I include -o test_model_moe tools/test_model_moe.c \
 *        src/gguf_reader.o src/wubu_ssm.o src/wubu_mobius.o \
 *        src/wubu_moe.o src/wubu_model.o src/wubu_tokenizer.o \
 *        src/qlearner.o src/dequant_iq2_xxs.o -lm -fopenmp
 * Usage: ./test_model_moe model.gguf
 * Compares library MoE (wubu_moe_forward) vs lazy_moe_decode (infer_text)
 */
#include "wubu_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, argv[1])) {
        fprintf(stderr, "Failed to load model\n"); return 1;
    }

    // Enable MoE
    mdl.enable_moe = true;

    // Single BOS token
    int token_ids[1] = { 248044 };
    int B = 1, T = 1;
    int64_t n_logits = (int64_t)B * T * mdl.vocab_size;
    float *logits = (float *)malloc(n_logits * sizeof(float));
    if (!logits) { fprintf(stderr, "OOM for logits\n"); return 1; }

    fprintf(stderr, "Running wubu_model_forward with MoE...\n");
    fflush(stderr);

    // DEBUG: dump MoE input at layer 0
    // We need to intercept the forward. Let's just do a quick comparison
    // by running one layer manually.
    
    wubu_model_forward(&mdl, token_ids, B, T, logits);

    // Dump logits
    FILE *f = fopen("/tmp/libmoe_logits.bin", "wb");
    if (f) {
        fwrite(logits, sizeof(float), mdl.vocab_size, f);
        fclose(f);
    }

    // Stats
    double sum = 0, sumsq = 0;
    for (int i = 0; i < 320 && i < mdl.vocab_size; i++) {
        sum += logits[i];
        sumsq += (double)logits[i] * logits[i];
    }
    fprintf(stderr, "Logits (first 320): mean=%.6f rms=%.6f\n",
            (float)(sum/320), (float)sqrt(sumsq/320));

    fflush(stderr);
    free(logits);
    wubu_model_free(&mdl);
    fprintf(stderr, "Done\n");
    return 0;
}
