/**
 * q5k_debug.c — Dump raw Q5_K block parameters for a specific tensor
 * Single file: copies needed functions from gguf_reader.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Copy needed inline functions from gguf_reader.c
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) {
        return f32_from_f16(sign, 0, mant);
    } else if (exp == 31) {
        return f32_from_f16(sign, 255, mant);
    } else {
        return f32_from_f16(sign, exp - 15 + 127, mant);
    }
}

#define QK_K 256
#define Q5_K_BLOCK_SIZE 176

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// GGML types needed
enum ggml_type {
    GGML_TYPE_F32    = 0,
    GGML_TYPE_Q5_K   = 13,
    GGML_TYPE_Q6_K   = 14,
};

// Minimal GGUF reader
#include "gguf_reader.h"

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
        : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *tname = argc > 2 ? argv[2] : "blk.0.attn_qkv.weight";

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);

    gguf_tensor_info *t = gguf_find_tensor(ctx, tname);
    if (!t) { printf("Tensor %s not found\n", tname); return 1; }

    int64_t n_elems = 1;
    for (int d = 0; d < t->n_dims; d++) n_elems *= t->dims[d];
    printf("Tensor: %s type=%d dims=%ld n_elems=%ld\n", tname, t->ggml_type, (long)t->dims[0], (long)n_elems);

    if (t->ggml_type != GGML_TYPE_Q5_K) {
        printf("Not Q5_K type. Dequantizing and dumping stats...\n");
        float *buf = (float *)malloc(n_elems * sizeof(float));
        int n = gguf_read_tensor_f32(ctx, t, buf, n_elems);
        printf("Dequant: %d elems\n", n);
        double mean=0, var=0;
        double mn=1e30, mx=-1e30;
        for (int i = 0; i < n && i < n_elems; i++) {
            mean += buf[i]; var += (double)buf[i]*buf[i];
            if (buf[i] < mn) mn = buf[i];
            if (buf[i] > mx) mx = buf[i];
        }
        mean /= n_elems;
        var = var/n_elems - mean*mean;
        double std = sqrt(var);
        // Histogram
        int bins[20] = {0};
        double bin_w = (mx - mn) / 20.0;
        if (bin_w > 0) {
            for (int i = 0; i < n_elems; i++) {
                int b = (int)((buf[i] - mn) / bin_w);
                if (b >= 20) b = 19;
                bins[b]++;
            }
        }
        printf("Stats: mean=%.6f std=%.6f min=%.6f max=%.6f\n", mean, std, mn, mx);
        printf("Histogram (%d bins):\n", 20);
        for (int i = 0; i < 20; i++) {
            float low = mn + i * bin_w;
            float high = low + bin_w;
            printf("  [%8.4f-%8.4f] %d\n", low, high, bins[i]);
        }
        printf("First 32: ");
        for (int i = 0; i < 32 && i < n_elems; i++) printf("%.4f ", buf[i]);
        printf("\n");
        free(buf);
        gguf_close(ctx);
        return 0;
    }

    // Q5_K: dump block parameters for first 8 blocks
    int64_t n_blocks = (n_elems + QK_K - 1) / QK_K;
    printf("Total blocks: %ld\n", (long)n_blocks);

    int show = n_blocks < 8 ? (int)n_blocks : 8;
    for (int64_t bi = 0; bi < show; bi++) {
        const uint8_t *block = (const uint8_t *)ctx->data_blob + t->data_offset + bi * Q5_K_BLOCK_SIZE;
        
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block, 2);
        memcpy(&dmin_bits, block + 2, 2);
        float d = f16_to_f32(d_bits);
        float dmin = f16_to_f32(dmin_bits);
        
        printf("Block %ld: d=%e dmin=%e\n", (long)bi, d, dmin);
        printf("  scales[0..11]:");
        for (int i = 0; i < 12; i++) printf(" %02x", block[4+i]);
        printf("\n");
        
        // Dequant this block
        float deq[256];
        uint8_t sc, m;
        int is = 0;
        const uint8_t *scales = block + 4;
        const uint8_t *qh = block + 16;
        const uint8_t *qs = block + 48;
        
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            float d1 = d * sc; float m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * sc; float m2 = dmin * m;
            
            int chunk_id = j / 64;
            int ql_base = j / 2;
            
            for (int l = 0; l < 32; l++) {
                uint8_t lo = qs[ql_base + l];
                int hi1 = (qh[l] >> (chunk_id * 2 + 0)) & 1;
                int hi2 = (qh[l] >> (chunk_id * 2 + 1)) & 1;
                float val1 = (float)((lo & 0x0F) + (hi1 ? 16 : 0));
                float val2 = (float)((lo >> 4)   + (hi2 ? 16 : 0));
                deq[j + l]      = d1 * val1 - m1;
                deq[j + 32 + l] = d2 * val2 - m2;
            }
            is += 2;
        }
        
        double bmean = 0, bvar = 0, bmn = 1e30, bmx = -1e30;
        for (int i = 0; i < QK_K; i++) {
            bmean += deq[i]; bvar += (double)deq[i]*deq[i];
            if (deq[i] < bmn) bmn = deq[i];
            if (deq[i] > bmx) bmx = deq[i];
        }
        bmean /= QK_K;
        bvar = bvar/QK_K - bmean*bmean;
        printf("  Deq: mean=%.6f std=%.6f min=%.6f max=%.6f\n",
               bmean, sqrt(bvar), bmn, bmx);
        printf("  First 16: ");
        for (int i = 0; i < 16; i++) printf("%.4f ", deq[i]);
        printf("\n");
    }

    gguf_close(ctx);
    return 0;
}
