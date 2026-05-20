#include "gguf_reader.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Same dequant as GPU code
static void dequant_block_q4_K_f32_test(const uint8_t *block, float *out) {
    const uint16_t d_half = *(const uint16_t*)(block);
    const uint16_t dmin_half = *(const uint16_t*)(block + 2);
    int sign = (d_half >> 15) & 1;
    int exp  = (d_half >> 10) & 0x1F;
    int mant = d_half & 0x3FF;
    float d;
    if (exp == 0) d = (float)mant / 1024.0f * 0.000061035f * (sign ? -1.0f : 1.0f);
    else if (exp == 31) d = sign ? -__builtin_huge_valf() : __builtin_huge_valf();
    else d = ldexpf(1.0f + (float)mant / 1024.0f, exp - 15) * (sign ? -1.0f : 1.0f);
    sign = (dmin_half >> 15) & 1;
    exp  = (dmin_half >> 10) & 0x1F;
    mant = dmin_half & 0x3FF;
    float dmin;
    if (exp == 0) dmin = (float)mant / 1024.0f * 0.000061035f * (sign ? -1.0f : 1.0f);
    else if (exp == 31) dmin = sign ? -__builtin_huge_valf() : __builtin_huge_valf();
    else dmin = ldexpf(1.0f + (float)mant / 1024.0f, exp - 15) * (sign ? -1.0f : 1.0f);
    const uint8_t *scales_raw = block + 4;
    uint32_t utmp[4];
    memcpy(utmp, scales_raw, 12);
    utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0f) | (((utmp[1] >> 6) & 0x03030303) << 4);
    const uint32_t uaux = utmp[1] & 0x3f3f3f3f;
    utmp[1] = (utmp[2] & 0x0f0f0f0f) | (((utmp[0] >> 6) & 0x03030303) << 4);
    utmp[2] = uaux;
    utmp[0] &= 0x3f3f3f3f;
    const uint8_t *scales = (const uint8_t*)&utmp[0];
    const uint8_t *mins   = (const uint8_t*)&utmp[2];
    const uint8_t *qs = block + 4 + 12;
    int idx = 0;
    for (int j = 0; j < 8; j++) {
        float dl = d * scales[j];
        float ml = dmin * mins[j/2];
        for (int k = 0; k < 32; k += 2) {
            int v0 = qs[idx/2] & 0xF;
            int v1 = qs[idx/2] >> 4;
            out[idx] = (float)v0 * dl - ml;
            out[idx+1] = (float)v1 * dl - ml;
            idx += 2;
        }
        if (j % 2 == 1) qs += 16;
    }
}

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;

    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    const uint8_t *weight_q = blob + t->data_offset;
    int D = 2048, V = 248320;
    const int QK_K = 256;
    const int blk_sz = 144;

    // Dequant first 3 blocks of row 0 using our formula
    float dequant[3 * QK_K];
    int blocks_per_row = (V + QK_K - 1) / QK_K;
    for (int b = 0; b < 3; b++) {
        dequant_block_q4_K_f32_test(weight_q + b * blk_sz, dequant + b * QK_K);
    }

    // Now dequant using gguf_dequantize for the first 3 blocks
    float ref[3 * QK_K];
    gguf_dequantize(weight_q, t->ggml_type, 3 * QK_K, ref);

    // Compare
    printf("Comparing GPU dequant vs gguf_dequantize (first 3 blocks = 768 elems):\n");
    double max_err = 0;
    int max_idx = -1;
    for (int i = 0; i < 3 * QK_K; i++) {
        double err = fabs(dequant[i] - ref[i]);
        if (err > max_err) { max_err = err; max_idx = i; }
    }
    printf("  Max error: %.10f at index %d\n", max_err, max_idx);
    printf("  GPU[%d]: %.10f  ref[%d]: %.10f\n", max_idx, dequant[max_idx], max_idx, ref[max_idx]);
    
    // Check first 8
    printf("\nFirst 8 values:\n");
    for (int i = 0; i < 8; i++)
        printf("  [%d] GPU=%.6f ref=%.6f diff=%.10f\n", i, dequant[i], ref[i], (double)(dequant[i]-ref[i]));

    // Also check: output.weight layout — is it [D,V] or [V,D]?
    // GGUF stores leading dim first: dims[0]=2048, dims[1]=248320
    // In our convention, dims[0] is the INNER dimension (elements per column in row-major)
    printf("\nTensor dims: [%lld, %lld]\n", (long long)t->dims[0], (long long)t->dims[1]);
    printf("In our code: D=%d (innermost/hidden), V=%d (outermost/vocab)\n", D, V);
    printf("Row stride in Q4_K blocks: %d blocks = %d bytes\n", blocks_per_row, blocks_per_row * blk_sz);

    gguf_close(ctx);
    return 0;
}
