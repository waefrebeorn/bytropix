#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

// GPU-style dequant matching gpu_output_proj.cu
static void dequant_block_q4_K_gpu(const uint8_t *block, float *out) {
    uint16_t d_bits, dmin_bits;
    memcpy(&d_bits, block, 2);
    memcpy(&dmin_bits, block + 2, 2);
    int s = (d_bits >> 15) & 1, e = (d_bits >> 10) & 0x1F, m = d_bits & 0x3FF;
    float d = (e == 0) ? ldexpf((float)m / 1024.0f, -14) * (s ? -1.0f : 1.0f)
            : (e == 31) ? (s ? -__builtin_huge_valf() : __builtin_huge_valf())
            : ldexpf(1.0f + (float)m / 1024.0f, e - 15) * (s ? -1.0f : 1.0f);
    s = (dmin_bits >> 15) & 1; e = (dmin_bits >> 10) & 0x1F; m = dmin_bits & 0x3FF;
    float dmin = (e == 0) ? ldexpf((float)m / 1024.0f, -14) * (s ? -1.0f : 1.0f)
               : (e == 31) ? (s ? -__builtin_huge_valf() : __builtin_huge_valf())
               : ldexpf(1.0f + (float)m / 1024.0f, e - 15) * (s ? -1.0f : 1.0f);
    const uint8_t *scales = block + 4;
    const uint8_t *qs = block + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc1, m1, sc2, m2;
        int idx = is;
        if (idx < 4) { sc1 = scales[idx] & 63; m1 = scales[idx + 4] & 63; }
        else { sc1 = (scales[idx+4] & 0xF) | ((scales[idx-4] >> 6) << 4);
               m1  = (scales[idx+4] >>  4) | ((scales[idx  ] >> 6) << 4); }
        idx = is + 1;
        if (idx < 4) { sc2 = scales[idx] & 63; m2 = scales[idx + 4] & 63; }
        else { sc2 = (scales[idx+4] & 0xF) | ((scales[idx-4] >> 6) << 4);
               m2  = (scales[idx+4] >>  4) | ((scales[idx  ] >> 6) << 4); }
        float d1 = d * (float)sc1; float ml1 = dmin * (float)m1;
        float d2 = d * (float)sc2; float ml2 = dmin * (float)m2;
        const uint8_t *bq = qs + j/2;
        for (int l = 0; l < 32; l++) out[j + l]      = d1 * (float)(bq[l] & 0xF) - ml1;
        for (int l = 0; l < 32; l++) out[j + 32 + l] = d2 * (float)(bq[l] >> 4) - ml2;
        is += 2;
    }
}

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;

    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    const uint8_t *wq = blob + t->data_offset;
    int blk_sz = 144;

    // Dequant first block with GPU method
    float gpu_blk[256];
    dequant_block_q4_K_gpu(wq, gpu_blk);

    // Dequant with reference
    float ref_blk[256];
    gguf_dequantize(wq, t->ggml_type, 256, ref_blk);

    // Compare
    double max_err = 0; int max_i = -1;
    for (int i = 0; i < 256; i++) {
        double err = fabs(gpu_blk[i] - ref_blk[i]);
        if (err > max_err) { max_err = err; max_i = i; }
    }
    printf("Block 0 comparison (256 elems):\n");
    printf("  Max error: %.10f at index %d\n", max_err, max_i);
    printf("  GPU[%d]=%.10f ref[%d]=%.10f\n", max_i, gpu_blk[max_i], max_i, ref_blk[max_i]);

    // Print sub-block boundaries
    for (int sb = 0; sb < 8; sb++) {
        int idx = sb * 32;
        printf("  Sub-block %d [%d]: GPU=%.6f ref=%.6f\n", sb, idx, gpu_blk[idx], ref_blk[idx]);
    }

    // Second block too
    float gpu2[256], ref2[256];
    dequant_block_q4_K_gpu(wq + blk_sz, gpu2);
    gguf_dequantize(wq + blk_sz, t->ggml_type, 256, ref2);
    max_err = 0; max_i = -1;
    for (int i = 0; i < 256; i++) {
        double err = fabs(gpu2[i] - ref2[i]);
        if (err > max_err) { max_err = err; max_i = i; }
    }
    printf("\nBlock 1 comparison:\n  Max error: %.10f at index %d\n", max_err, max_i);

    // Skip to block 486 (halfway through row 0, ~124K elements)
    float gpu3[256], ref3[256];
    int mid_block = 485;
    dequant_block_q4_K_gpu(wq + mid_block * blk_sz, gpu3);
    gguf_dequantize(wq + mid_block * blk_sz, t->ggml_type, 256, ref3);
    max_err = 0; max_i = -1;
    for (int i = 0; i < 256; i++) {
        double err = fabs(gpu3[i] - ref3[i]);
        if (err > max_err) { max_err = err; max_i = i; }
    }
    printf("\nBlock %d (mid-row):\n  Max error: %.10f at index %d\n", mid_block, max_err, max_i);

    gguf_close(ctx);
    return 0;
}
