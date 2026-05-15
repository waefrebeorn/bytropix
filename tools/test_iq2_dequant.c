/**
 * test_iq2_dequant.c — IQ2_XXS dequant verification.
 *
 * Loads blk.0.ffn_gate_exps.weight from a GGUF model, dequantizes using both
 * our gguf_dequantize and an independent reference implementation matching
 * llama.cpp's dequantize_row_iq2_xxs, then compares element-by-element.
 *
 * NOTE: The fp16-to-fp32 conversion MUST match gguf_reader.c's f16_to_f32,
 * particularly for subnormals (exp=0). See f16_to_f32 in gguf_reader.c for
 * the correct subnormal handling (normal_val - 0x1p-14f).
 *
 * Usage: test_iq2_dequant <model.gguf>
 * Compile: make test_iq2_dequant
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define QK_K 256
#define IQ2_XXS_BLOCK_SIZE 66

// ================================================================
// IQ2_XXS grid/sign tables (matching llama.cpp ggml-common.h)
// ================================================================
static const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static const uint8_t kmask_iq2xs_ref[8] = {1, 2, 4, 8, 16, 32, 64, 128};
static const uint8_t ksigns_iq2xs_ref[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

// ================================================================
// fp16-to-fp32: MUST match gguf_reader.c's f16_to_f32 exactly.
// gguf_reader.c uses a clever trick for subnormals (exp=0):
//   normal_val = 2^(-14) * (1 + mant/2^23)  [treat as normal with exp=1]
//   result = normal_val - 2^(-14)           [remove leading 1]
// This is equivalent to: mant/1024 * 2^(-14)
// ================================================================
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) {
        uint32_t normal_f32 = (sign << 31) | ((1 + 112) << 23) | (mant << 13);
        float normal_val;
        memcpy(&normal_val, &normal_f32, 4);
        return normal_val - 0x1p-14f;
    }
    uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f32, 4);
    return result;
}

// ================================================================
// Reference dequant: matches llama.cpp dequantize_row_iq2_xxs
// ================================================================
static void dequantize_iq2_xxs_reference(const uint8_t *data, float *output, int64_t n_elems) {
    int64_t n_blocks = (n_elems + QK_K - 1) / QK_K;
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;

    for (int64_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = data + b * IQ2_XXS_BLOCK_SIZE;
        uint16_t d_bits;
        memcpy(&d_bits, block, 2);
        float d = fp16_to_fp32(d_bits);
        const uint16_t *qs16 = (const uint16_t *)(block + 2);

        for (int ib32 = 0; ib32 < QK_K/32; ib32++) {
            memcpy(aux32, qs16 + 4*ib32, 2*sizeof(uint32_t));
            float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;

            for (int l = 0; l < 4; l++) {
                const uint8_t *grid = (const uint8_t *)(&iq2xxs_grid[aux8[l]]);
                uint8_t signs = ksigns_iq2xs_ref[(aux32[1] >> (7*l)) & 127];
                int64_t base = b * QK_K + ib32 * 32 + l * 8;

                for (int j = 0; j < 8; j++) {
                    if (base + j >= n_elems) break;
                    float val = db * (float)grid[j];
                    output[base + j] = (signs & kmask_iq2xs_ref[j]) ? -val : val;
                }
            }
        }
    }
}

// ================================================================
// Main
// ================================================================
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) return 1;

    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.ffn_gate_exps.weight");
    if (!t) {
        fprintf(stderr, "Tensor 'blk.0.ffn_gate_exps.weight' not found\n");
        gguf_close(ctx);
        return 1;
    }

    int64_t n_elems = 1;
    for (int d = 0; d < t->n_dims; d++) n_elems *= t->dims[d];
    int64_t raw_size = gguf_raw_size(t->ggml_type, n_elems);

    printf("=== IQ2_XXS Dequant Verification ===\n");
    printf("Tensor: %s\n", t->name);
    printf("  dims=[%ld,%ld,%ld]  type=%d (IQ2_XXS)\n",
           (long)t->dims[0], (long)t->dims[1], (long)t->dims[2], t->ggml_type);
    printf("  elements=%ld  raw_bytes=%ld\n", (long)n_elems, (long)raw_size);

    // Buffer the data blob for raw access
    if (!ctx->data_blob && !gguf_buffer_data(ctx)) {
        fprintf(stderr, "Failed to buffer data\n");
        gguf_close(ctx);
        return 1;
    }

    // Get raw quantized data
    const uint8_t *raw_data = (const uint8_t *)ctx->data_blob + t->data_offset;

    // Dequantize using our gguf_dequantize
    float *our_output = (float *)malloc(n_elems * sizeof(float));
    if (!our_output) { perror("malloc"); gguf_close(ctx); return 1; }
    gguf_dequantize(raw_data, t->ggml_type, n_elems, our_output);

    // Dequantize using reference implementation
    float *ref_output = (float *)malloc(n_elems * sizeof(float));
    if (!ref_output) { free(our_output); gguf_close(ctx); return 1; }
    dequantize_iq2_xxs_reference(raw_data, ref_output, n_elems);

    // Compare element-by-element
    int64_t n_diff = 0;
    float max_diff = 0.0f;
    int64_t first_idx[10];
    float first_our[10], first_ref[10];
    int n_first = 0;

    for (int64_t i = 0; i < n_elems; i++) {
        float diff = fabsf(our_output[i] - ref_output[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-6f) {
            if (n_first < 10) {
                first_idx[n_first] = i;
                first_our[n_first] = our_output[i];
                first_ref[n_first] = ref_output[i];
                n_first++;
            }
            n_diff++;
        }
    }

    // Print results
    printf("\n=== Results ===\n");
    printf("Total elements compared: %ld\n", (long)n_elems);
    printf("Max absolute difference: %e\n", max_diff);
    printf("Differing elements (>1e-6): %ld\n", (long)n_diff);

    if (n_diff == 0) {
        printf("\n✓ PASS: Our IQ2_XXS dequant matches llama.cpp reference.\n");
    } else {
        printf("\nFirst %d differing pairs:\n", n_first < 10 ? n_first : 10);
        for (int k = 0; k < n_first; k++) {
            int64_t idx = first_idx[k];
            int64_t block = idx / 256, off = idx % 256;
            printf("  idx=%10ld (blk=%ld,off=%ld)  our=%.8f  ref=%.8f  diff=%e\n",
                   (long)idx, (long)block, (long)off,
                   first_our[k], first_ref[k], fabsf(first_our[k] - first_ref[k]));
        }
        printf("\n✗ FAIL: Dequant mismatch detected.\n");
    }

    // Print summary statistics
    double sum = 0, sum2 = 0;
    float vmin = our_output[0], vmax = our_output[0];
    for (int64_t i = 0; i < n_elems; i++) {
        sum += our_output[i];
        sum2 += (double)our_output[i] * our_output[i];
        if (our_output[i] < vmin) vmin = our_output[i];
        if (our_output[i] > vmax) vmax = our_output[i];
    }
    printf("\n=== Summary Stats ===\n");
    printf("  Range:   [%.4f, %.4f]\n", vmin, vmax);
    printf("  Mean:    %.6f\n", sum / n_elems);
    printf("  StdDev:  %.6f\n", sqrt(sum2 / n_elems - (sum / n_elems) * (sum / n_elems)));
    printf("  Nonzero: %ld / %ld\n", (long)n_diff, (long)n_elems);

    free(our_output);
    free(ref_output);
    gguf_close(ctx);
    return (n_diff == 0) ? 0 : 1;
}
