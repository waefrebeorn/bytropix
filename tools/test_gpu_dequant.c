#include "gguf_reader.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Copy of the GPU dequant function for CPU test
static void dequant_block_q4_K_f32_cpu(const uint8_t *block, float *out) {
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
    if (!ctx) { fprintf(stderr, "Failed to open\n"); return 1; }
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;

    // Find output.weight
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    if (!t) { fprintf(stderr, "output.weight not found\n"); return 1; }
    printf("output.weight: type=%d dims=%lld %lld\n", t->ggml_type,
           (long long)t->dims[0], (long long)t->dims[1]);

    const uint8_t *weight_q = blob + t->data_offset;
    int D = 2048, V = 248320;
    const int QK_K = 256;
    const int blk_sz_q4k = 144;

    // Dequant first block of first row
    float block_out[QK_K];
    printf("\nFirst Q4_K block of output.weight (row 0, cols 0-255):\n");
    dequant_block_q4_K_f32_cpu(weight_q, block_out);
    for (int i = 0; i < 8; i++)
        printf("  [%d] = %.6f\n", i, block_out[i]);
    printf("  ...\n");
    for (int i = 248; i < 256; i++)
        printf("  [%d] = %.6f\n", i, block_out[i]);

    // Also compare: compute via gguf_dequantize for first row
    printf("\nUsing gguf_raw_size for row 0: %lld bytes\n",
           (long long)gguf_raw_size(t->ggml_type, V));
    
    // Check total memory for GPU dequant
    int64_t n_blocks = (int64_t)D * V / QK_K;
    printf("Total Q4_K blocks: %lld\n", (long long)n_blocks);
    printf("F32 weight size: %.1f MB\n", (double)D * V * 4 / 1048576.0);
    printf("Quantized size: %lld bytes (%.1f MB)\n",
           (long long)n_blocks * blk_sz_q4k,
           (double)n_blocks * blk_sz_q4k / 1048576.0);

    gguf_close(ctx);
    return 0;
}
