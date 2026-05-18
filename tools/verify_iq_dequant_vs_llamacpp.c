/**
 * verify_iq_dequant_vs_llamacpp.c
 *
 * Compare bytropix IQ2_XXS/IQ3_XXS dequant against llama.cpp's dequant.
 * Links against libggml-base.so (which exports dequant functions).
 *
 * Build:
 *   cd /home/wubu/bytropix
 *   gcc -O2 -o verify_iq_dequant tools/verify_iq_dequant_vs_llamacpp.c \
 *       src/gguf_reader.o src/dequant_iq2_xxs.o \
 *       -I /home/wubu/llama.cpp/ggml/src \
 *       -I /home/wubu/llama.cpp/ggml/include \
 *       -L /home/wubu/llama.cpp/build/bin -lggml-base \
 *       -lm -Wl,-rpath,/home/wubu/llama.cpp/build/bin
 *
 * Run: ./verify_iq_dequant
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Only use llama.cpp headers (not bytropix's gguf_reader.h which duplicates the enum)
#include "ggml-common.h"
#include "ggml-quants.h"

// Forward-declare bytropix dequant functions (defined in src/gguf_reader.o and src/dequant_iq2_xxs.o)
void dequantize_iq2_xxs_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq3_xxs_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq4_xs_row(const uint8_t *data, float *output, int64_t n_elems);

#define QK_K 256

static int test_iq_type(
    const char *name,
    int n_blocks,
    int block_size,
    void (*bytropix_dequant)(const uint8_t *, float *, int64_t),
    void (*llamacpp_dequant)(const void *, float *, int64_t))
{
    int total_elems = n_blocks * QK_K;
    int total_bytes = n_blocks * block_size;

    uint8_t *raw = (uint8_t *)malloc(total_bytes);
    float *bytropix_out = (float *)malloc(total_elems * sizeof(float));
    float *llamacpp_out = (float *)malloc(total_elems * sizeof(float));

    if (!raw || !bytropix_out || !llamacpp_out) {
        printf("  FAIL: allocation error\n");
        free(raw); free(bytropix_out); free(llamacpp_out);
        return 0;
    }

    // Generate semi-realistic quantized data with valid fp16 scale
    srand(42);
    for (int b = 0; b < n_blocks; b++) {
        uint8_t *block = raw + b * block_size;
        // Scale: fp16 between 0.1 and 10.0
        float scale_f32 = 0.1f + (float)(rand() % 1000) / 100.0f;
        uint32_t f32_bits;
        memcpy(&f32_bits, &scale_f32, 4);
        uint16_t scale_f16 = ((f32_bits >> 16) & 0x8000) |
                             ((((f32_bits >> 23) - 127 + 15) & 0x1F) << 10) |
                             ((f32_bits >> 13) & 0x03FF);
        block[0] = scale_f16 & 0xFF;
        block[1] = (scale_f16 >> 8) & 0xFF;
        for (int k = 2; k < block_size; k++) {
            block[k] = (uint8_t)(rand() & 0xFF);
        }
    }

    // === bytropix dequant ===
    bytropix_dequant(raw, bytropix_out, total_elems);

    // === llama.cpp dequant ===
    llamacpp_dequant(raw, llamacpp_out, total_elems);

    // === Compare ===
    double max_err = 0.0;
    double total_err = 0.0;
    int n_bad = 0;
    double ref_max_abs = 0.0;

    for (int i = 0; i < total_elems; i++) {
        double err = fabs((double)bytropix_out[i] - (double)llamacpp_out[i]);
        double ref_abs = fabs((double)llamacpp_out[i]);
        if (ref_abs > ref_max_abs) ref_max_abs = ref_abs;
        if (err > max_err) max_err = err;
        total_err += err;
        double tol = 1e-4 * fmax(1.0, ref_max_abs);
        if (err > tol) {
            n_bad++;
            if (n_bad <= 10) {
                printf("  MISMATCH[%d]: bytropix=%.10f  llamacpp=%.10f  diff=%.10f\n",
                       i, bytropix_out[i], llamacpp_out[i], err);
            }
        }
    }

    double avg_err = total_err / total_elems;
    printf("\n  %s: %d blocks (%d elems), block_size=%d\n", name, n_blocks, total_elems, block_size);
    printf("    ref_max_abs = %.6f\n", ref_max_abs);
    printf("    max_abs_err = %.10f\n", max_err);
    printf("    avg_abs_err = %.10f\n", avg_err);
    printf("    mismatches  = %d / %d\n", n_bad, total_elems);

    printf("    bytropix[0..7]: ");
    for (int i = 0; i < 8 && i < total_elems; i++) printf("%+.6f ", bytropix_out[i]);
    printf("\n    llamacpp[0..7]: ");
    for (int i = 0; i < 8 && i < total_elems; i++) printf("%+.6f ", llamacpp_out[i]);
    printf("\n");

    int pass = (n_bad == 0);
    printf("    %s\n\n", pass ? "PASS" : "FAIL");

    free(raw);
    free(bytropix_out);
    free(llamacpp_out);
    return pass;
}

int main(void) {
    int passed = 0, failed = 0;
    int n_blocks = 10;

    printf("=== IQ Dequant: bytropix vs llama.cpp (libggml-base) ===\n");

    // IQ2_XXS: block_size=66
    if (test_iq_type("IQ2_XXS", n_blocks, sizeof(block_iq2_xxs),
                     dequantize_iq2_xxs_row,
                     (void (*)(const void *, float *, int64_t))dequantize_row_iq2_xxs))
        passed++; else failed++;

    // IQ3_XXS: block_size=98
    if (test_iq_type("IQ3_XXS", n_blocks, sizeof(block_iq3_xxs),
                     dequantize_iq3_xxs_row,
                     (void (*)(const void *, float *, int64_t))dequantize_row_iq3_xxs))
        passed++; else failed++;

    // IQ4_XS: block_size=136
    if (test_iq_type("IQ4_XS", n_blocks, sizeof(block_iq4_xs),
                     dequantize_iq4_xs_row,
                     (void (*)(const void *, float *, int64_t))dequantize_row_iq4_xs))
        passed++; else failed++;

    printf("========================================\n");
    printf("Results: %d passed, %d failed out of %d\n",
           passed, failed, passed + failed);
    printf("========================================\n");

    return failed > 0 ? 1 : 0;
}
