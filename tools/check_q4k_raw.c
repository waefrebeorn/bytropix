#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Copy the dequant inline with debug prints
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4); }
}

int main() {
    FILE *f = fopen("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
    // Skip to output.weight data
    // Data blob at offset 10990048
    // output.weight data_offset from tensor header
    
    // Simpler: just read from the gguf_reader directly
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    
    // Read raw data for first 10 blocks
    int n_blocks = 10;
    int raw_size = n_blocks * 176;
    uint8_t *raw = malloc(raw_size);
    
    fseek(ctx->file, ctx->data_blob_offset + t->data_offset, SEEK_SET);
    fread(raw, 1, raw_size, ctx->file);
    
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *block = raw + b * 176;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block, 2);
        memcpy(&dmin_bits, block + 2, 2);
        float d = f16_to_f32(d_bits);
        float dmin = f16_to_f32(dmin_bits);
        
        printf("Block %d: d=%f dmin=%f\n", b, d, dmin);
        
        // Print scales
        const uint8_t *scales = block + 4;
        printf("  scales[0:11]:");
        for (int i = 0; i < 12; i++) printf(" %02x", scales[i]);
        printf("\n");
        
        // Check get_scale_min_k4 for groups 0-7
        for (int is = 0; is < 8; is += 2) {
            uint8_t sc, m;
            get_scale_min_k4(is, scales, &sc, &m);
            printf("  group %d/%d: sc=%u m=%u d*sc=%f dmin*m=%f\n", is/2, is/2+1, sc, m, d*sc, dmin*m);
        }
        
        // Dequant and check stats for this block
        float vals[256];
        int64_t n_elems = 256;
        // Manual dequant for this block
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            float d1 = d * sc; float m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * sc; float m2 = dmin * m;
            
            const uint8_t *bq = block + 48 + j/2;
            for (int l = 0; l < 32; l++)
                vals[b*256 + j + l] = d1 * (bq[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++)
                vals[b*256 + j + 32 + l] = d2 * (bq[l] >> 4) - m2;
            
            is += 2;
        }
        
        double mean = 0, min_v = 1e30, max_v = -1e30;
        for (int i = 0; i < 256; i++) { mean += vals[i]; if(vals[i]<min_v)min_v=vals[i]; if(vals[i]>max_v)max_v=vals[i]; }
        mean /= 256;
        printf("  dequant: mean=%.4f min=%.4f max=%.4f\n", mean, min_v, max_v);
        printf("  first 8: ");
        for (int i = 0; i < 8; i++) printf(" %.4f", vals[i]);
        printf("\n\n");
    }
    
    free(raw);
    gguf_close(ctx);
    fclose(f);
    return 0;
}
