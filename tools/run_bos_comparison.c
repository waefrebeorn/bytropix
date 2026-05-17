/**
 * run_bos_comparison.c — Run our MoE path with BOS token and compare with ref.
 * Build: gcc -O2 -I include -o run_bos_comparison tools/run_bos_comparison.c \
 *        src/gguf_reader.o src/wubu_ssm.o src/wubu_mobius.o \
 *        src/wubu_moe.o src/wubu_model.o src/wubu_tokenizer.o \
 *        src/qlearner.o src/dequant_iq2_xxs.o -lm -fopenmp
 * Usage: ./run_bos_comparison model.gguf
 */
#include "wubu_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }
    
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, argv[1])) return 1;
    
    // Enable MoE
    mdl.enable_moe = true;
    
    // Single BOS token (same as reference)
    int token_ids[1] = { 248044 };
    int B = 1, T = 1;
    int64_t n_logits = (int64_t)B * T * mdl.vocab_size;
    float *logits = (float *)malloc(n_logits * sizeof(float));
    
    fprintf(stderr, "Running wubu_model_forward with MoE (BOS input)...\n");
    wubu_model_forward(&mdl, token_ids, B, T, logits);
    
    // Dump logits
    FILE *f = fopen("/tmp/our_bos_logits.bin", "wb");
    fwrite(logits, sizeof(float), mdl.vocab_size, f);
    fclose(f);
    
    // Stats
    double sum = 0, sumsq = 0;
    for (int i = 0; i < 320 && i < mdl.vocab_size; i++) {
        sum += logits[i];
        sumsq += (double)logits[i] * logits[i];
    }
    fprintf(stderr, "Our logits (first 320): mean=%.6f rms=%.6f\n",
            (float)(sum/320), (float)sqrt(sumsq/320));
    
    // Also get hidden state
    // wubu_model_forward doesn't expose hidden state, but we can
    // compare logits directly
    
    free(logits);
    wubu_model_free(&mdl);
    fprintf(stderr, "Done\n");
    return 0;
}
