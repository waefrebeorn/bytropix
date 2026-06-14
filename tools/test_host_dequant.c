#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Copy of the host dequant from gpu_gemma4_forward.cu */
static void host_dequant_q4k(const uint8_t *data, float *output, int64_t n_elems, int K) {
    #define LOCAL_Q4K_BLOCK_SIZE 144
    #define LOCAL_Q4K_N_ELEMS 256
    #define LOCAL_QK_K 256

    int64_t n_blocks = n_elems / LOCAL_QK_K;
    for (int64_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = data + b * LOCAL_Q4K_BLOCK_SIZE;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block, 2);
        memcpy(&dmin_bits, block + 2, 2);
        uint32_t d_sign = (d_bits >> 15) & 1, d_e = (d_bits >> 10) & 0x1F, d_m = d_bits & 0x03FF;
        float d;
        if (d_e == 0) {
            uint32_t normal_f32 = (d_sign << 31) | ((1 + 112) << 23) | (d_m << 13);
            float normal_val; memcpy(&normal_val, &normal_f32, 4);
            d = d_sign ? normal_val + 6.103515625e-5f : normal_val - 6.103515625e-5f;
        } else if (d_e == 31) {
            uint32_t f32 = (d_sign << 31) | (0xFF << 23) | (d_m << 13);
            memcpy(&d, &f32, 4);
        } else {
            uint32_t f32 = (d_sign << 31) | ((d_e + 112) << 23) | (d_m << 13);
            memcpy(&d, &f32, 4);
        }
        uint32_t dmin_sign = (dmin_bits >> 15) & 1, dmin_e = (dmin_bits >> 10) & 0x1F, dmin_m = dmin_bits & 0x03FF;
        float dmin;
        if (dmin_e == 0) {
            uint32_t normal_f32 = (dmin_sign << 31) | ((1 + 112) << 23) | (dmin_m << 13);
            float normal_val; memcpy(&normal_val, &normal_f32, 4);
            dmin = dmin_sign ? normal_val + 6.103515625e-5f : normal_val - 6.103515625e-5f;
        } else if (dmin_e == 31) {
            uint32_t f32 = (dmin_sign << 31) | (0xFF << 23) | (dmin_m << 13);
            memcpy(&dmin, &f32, 4);
        } else {
            uint32_t f32 = (dmin_sign << 31) | ((dmin_e + 112) << 23) | (dmin_m << 13);
            memcpy(&dmin, &f32, 4);
        }

        const uint8_t *scales = block + 4;
        const uint8_t *qs = block + 16;
        int is = 0;
        for (int j = 0; j < LOCAL_QK_K; j += 64) {
            uint8_t sc, m;
            if (is < 4) { sc = scales[is] & 63; m = scales[is + 4] & 63; }
            else { sc = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4); m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4); }
            if (b == 0 && j == 0) printf("  DBG host: is=%d, sc=%d, m=%d\n", is, sc, m);
            float d1 = d * sc; float m1 = dmin * m;
            if (is + 1 < 4) { sc = scales[is + 1] & 63; m = scales[is + 1 + 4] & 63; }
            else { sc = (scales[is + 1 + 4] & 0xF) | ((scales[is + 1 - 4] >> 6) << 4); m = (scales[is + 1 + 4] >> 4) | ((scales[is + 1] >> 6) << 4); }
            if (b == 0 && j == 0) printf("  DBG host: is+1=%d, sc=%d, m=%d\n", is+1, sc, m);
            float d2 = d * sc; float m2 = dmin * m;
            const uint8_t *bq = qs + j/2;
            int base = b * LOCAL_QK_K + j;
            for (int l = 0; l < 32; l++) {
                output[base + l] = d1 * (float)(bq[l] & 0xF) - m1;
                if (b == 0 && j == 0 && l < 4) {
                    printf("  DBG host: bq[%d]=0x%02x, nib=%d, d1=%.4f, m1=%.4f, out=%.4f\n", 
                        l, bq[l], bq[l] & 0xF, d1, m1, output[base + l]);
                }
            }
            for (int l = 0; l < 32; l++) {
                output[base + 32 + l] = d2 * (float)(bq[l] >> 4) - m2;
            }
            is += 2;
        }
    }
}

int main() {
    const char *path = "/home/wubu/models/gemma4/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open\n"); return 1; }
    gguf_buffer_data(ctx);
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;

    /* Find first layer's attn_q weight */
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.attn_q.weight");
    if (!t) { fprintf(stderr, "blk.0.attn_q.weight not found\n"); return 1; }
    printf("attn_q: type=%d dims=%lld %lld\n", t->ggml_type, (long long)t->dims[0], (long long)t->dims[1]);

    const uint8_t *weight_q = blob + t->data_offset;
    const int QK_K = 256;
    const int blk_sz_q4k = 144;

    /* Dequant first 3 blocks using host dequant (from gpu_gemma4_forward.cu) */
    float host_deq[3 * LOCAL_QK_K];
    int blocks_per_row = (t->dims[0] + LOCAL_QK_K - 1) / LOCAL_QK_K;  // K dimension is dims[0]
    printf("K=%lld, blocks_per_row=%d\n", t->dims[0], blocks_per_row);
    for (int b = 0; b < 3; b++) {
        host_dequant_q4k(weight_q + b * blk_sz_q4k, host_deq + b * LOCAL_QK_K, LOCAL_QK_K, t->dims[0]);
    }

    /* Dequant with reference gguf_dequantize */
    float ref[3 * LOCAL_QK_K];
    gguf_dequantize(weight_q, t->ggml_type, 3 * LOCAL_QK_K, ref);

    /* Compare */
    printf("\nComparing host_dequant vs gguf_dequantize (first 3 blocks = 768 elems):\n");
    double max_err = 0;
    int max_idx = -1;
    for (int i = 0; i < 3 * QK_K; i++) {
        double err = fabs(host_deq[i] - ref[i]);
        if (err > max_err) { max_err = err; max_idx = i; }
    }
    printf("  Max error: %.10f at index %d\n", max_err, max_idx);
    printf("  host[%d]: %.10f  ref[%d]: %.10f\n", max_idx, host_deq[max_idx], max_idx, ref[max_idx]);

    printf("\nFirst 16 values:\n");
    for (int i = 0; i < 16; i++)
        printf("  [%d] host=%.6f ref=%.6f diff=%.10f\n", i, host_deq[i], ref[i], (double)(host_deq[i]-ref[i]));

    gguf_close(ctx);
    return 0;
}
