/**
 * test_moe_layer0.c — Compare our MoE forward vs ggml for layer 0.
 * Build: gcc -O2 -I include -o test_moe_layer0 tools/test_moe_layer0.c \
 *        src/gguf_reader.o src/wubu_ssm.o src/wubu_mobius.o \
 *        src/wubu_moe.o src/wubu_model.o src/wubu_tokenizer.o \
 *        src/qlearner.o src/dequant_iq2_xxs.o -lm -fopenmp
 * Usage: ./test_moe_layer0 model.gguf
 */
#include "wubu_model.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }

    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { return 1; }

    // Load layer 0 MoE weights (F32 dequantized)
    moe_weights_t moe;
    if (!wubu_moe_load_layer(ctx, 0, &moe)) {
        fprintf(stderr, "Failed to load MoE layer 0\n");
        gguf_close(ctx);
        return 1;
    }
    fprintf(stderr, "MoE layer 0 loaded\n");

    // Create a simple input (all 1s)
    float x[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) x[i] = 1.0f;

    // Call our MoE forward
    float our_out[D_MODEL];
    wubu_moe_forward(x, 1, 1, &moe, our_out, NULL);

    // Dump for comparison
    FILE *f = fopen("/tmp/moe_layer0_out.bin", "wb");
    fwrite(our_out, sizeof(float), D_MODEL, f);
    fclose(f);

    double s = 0;
    for (int i = 0; i < D_MODEL; i++) s += (double)our_out[i] * our_out[i];
    fprintf(stderr, "Our MoE output: rms=%.6f\n", sqrt(s/D_MODEL));
    fprintf(stderr, "First 8 values: ");
    for (int i = 0; i < 8 && i < D_MODEL; i++) fprintf(stderr, "%.6f ", our_out[i]);
    fprintf(stderr, "\n");

    // Now compare with infer_text's lazy_moe_decode
    // We need to feed the same input through infer_text with MOE=1
    // and DUMP_LAYER_DIR to get layer 0 post-MoE
    // Then compare: our_out vs (infer_text L0 post-MoE - infer_text L0 post-attn)
    // Actually, infer_text dumps the RESIDUAL, not the MoE output
    // MoE output = post-MoE residual - post-attn residual
    // Let's load those dumps

    wubu_moe_free_layer(&moe);
    gguf_close(ctx);
    return 0;
}
