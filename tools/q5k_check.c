/**
 * Check Q5_K dequant by dumping block-level d/dmin for first few blocks
 * Includes gguf_reader.c for static function access
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// Minimal GGUF reader - just the part we need
#include "gguf_reader.h"

// Concat gguf_reader.c inline so we can call its static functions
#include "/home/wubu/bytropix/src/gguf_reader.c"

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
    int dtype = t->ggml_type;
    printf("Tensor: %s type=%d n_elems=%ld\n", tname, dtype, (long)n_elems);

    if (dtype == GGML_TYPE_Q5_K) {
        int64_t n_blocks = (n_elems + QK_K - 1) / QK_K;
        printf("Q5_K: %ld blocks\n", (long)n_blocks);
        int show = n_blocks < 8 ? (int)n_blocks : 8;
        
        for (int64_t bi = 0; bi < show; bi++) {
            const uint8_t *block = (const uint8_t *)ctx->data_blob + t->data_offset + bi * Q5_K_BLOCK_SIZE;
            
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits, block, 2);
            memcpy(&dmin_bits, block + 2, 2);
            float d = f16_to_f32(d_bits);
            float dmin = f16_to_f32(dmin_bits);
            
            printf("\nBlock %ld: d=%+e dmin=%+e\n", (long)bi, d, dmin);
            printf("  d_bits=0x%04x dmin_bits=0x%04x\n", d_bits, dmin_bits);
            
            // scales 6-bit decode
            printf("  scales=");
            for (int i = 0; i < 12; i++) printf(" %02x(%d)", block[4+i], block[4+i] & 63);
            printf("\n");
            
            // Decode sub-block scales
            for (int j = 0; j < 4; j++) {
                uint8_t sc, m;
                get_scale_min_k4(j, block + 4, &sc, &m);
                printf("  group %d: sc=%d m=%d d*sc=%e dmin*m=%e\n", j, sc, m, d*sc, dmin*m);
            }
            
            // Dequant this block using official function
            float deq[QK_K];
            dequantize_q5_K_row(block, deq, QK_K);
            
            double mean = 0, minv = 1e30, maxv = -1e30;
            for (int i = 0; i < QK_K; i++) {
                mean += deq[i];
                if (deq[i] < minv) minv = deq[i];
                if (deq[i] > maxv) maxv = deq[i];
            }
            mean /= QK_K;
            printf("  dequantized: mean=%+.6f min=%+.6f max=%+.6f\n", mean, minv, maxv);
            printf("  first 8: ");
            for (int i = 0; i < 8; i++) printf("%+.6f ", deq[i]);
            printf("\n");
        }
        
        // Also dump stats for ALL dequantized values
        float *all = (float *)malloc(n_elems * sizeof(float));
        dequantize_q5_K_row((const uint8_t *)ctx->data_blob + t->data_offset, all, n_elems);
        double am = 0, av = 0, amn = 1e30, amx = -1e30;
        int abins[20] = {0};
        for (int64_t i = 0; i < n_elems; i++) {
            am += all[i]; av += (double)all[i]*all[i];
            if (all[i] < amn) amn = all[i];
            if (all[i] > amx) amx = all[i];
        }
        am /= n_elems; av = av/n_elems - am*am;
        double bw = (amx - amn) / 20.0;
        if (bw > 0) {
            for (int64_t i = 0; i < n_elems; i++) {
                int b = (int)((all[i] - amn) / bw);
                if (b >= 20) b = 19;
                abins[b]++;
            }
        }
        printf("\n=== Full tensor stats ===\n");
        printf("mean=%+.6f std=%.6f min=%+.6f max=%+.6f\n", am, sqrt(av), amn, amx);
        printf("Histogram:\n");
        for (int i = 0; i < 20; i++) {
            printf("  [%+.6f-%+.6f] %d\n", amn+i*bw, amn+(i+1)*bw, abins[i]);
        }
        free(all);
        
    } else {
        // For non-Q5_K, just dequant and dump stats
        float *all = (float *)malloc(n_elems * sizeof(float));
        gguf_read_tensor_f32(ctx, t, all, n_elems);
        double am = 0, amn = 1e30, amx = -1e30;
        for (int64_t i = 0; i < n_elems; i++) {
            am += all[i];
            if (all[i] < amn) amn = all[i];
            if (all[i] > amx) amx = all[i];
        }
        am /= n_elems;
        printf("Stats: mean=%+.6f min=%+.6f max=%+.6f\n", am, amn, amx);
        printf("First 8: ");
        for (int i = 0; i < 8; i++) printf("%+.6f ", all[i]);
        printf("\n");
        free(all);
    }

    gguf_close(ctx);
    return 0;
}
