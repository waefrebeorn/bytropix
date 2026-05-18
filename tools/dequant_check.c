/* Dequant first Q4_K block of output.weight and save for Python comparison */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gguf_reader.h"

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    int type = t->ggml_type;
    printf("output.weight type=%d\n", type);
    
    // Read raw data of first block
    const uint8_t *data = (const uint8_t*)ctx->data_blob + t->data_offset;
    
    // Q4_K block: d(2) + dmin(2) + scales(12) + qs(128) = 144 bytes
    // Dump first 144 bytes as hex
    printf("First 144 bytes of output.weight (first Q4_K block):\n");
    for (int i = 0; i < 144; i += 16) {
        printf("  %03d: ", i);
        for (int j = 0; j < 16 && i+j < 144; j++)
            printf("%02x ", data[i+j]);
        printf("\n");
    }
    
    // Now let's dequant this block manually
    // Block layout: d(fp16,2) + dmin(fp16,2) + scales(12) + qs(128)
    uint16_t d_bits, dmin_bits;
    memcpy(&d_bits, data, 2);
    memcpy(&dmin_bits, data + 2, 2);
    
    // f16 to f32
    float d = f16_to_f32(d_bits);
    float dmin = f16_to_f32(dmin_bits);
    printf("d = %.10f (0x%04x), dmin = %.10f (0x%04x)\n", d, d_bits, dmin, dmin_bits);
    
    const uint8_t *scales = data + 4;
    printf("scales: ");
    for (int i = 0; i < 12; i++) printf("%02x ", scales[i]);
    printf("\n");
    
    const uint8_t *qs = data + 16;
    
    // Dequantize first group (elements 0-63)
    uint8_t sc, m;
    get_scale_min_k4(0, scales, &sc, &m);
    printf("Group 0: sc=%d m=%d\n", sc, m);
    float d1 = d * sc;
    float m1 = dmin * m;
    
    get_scale_min_k4(1, scales, &sc, &m);
    printf("Group 1: sc=%d m=%d\n", sc, m);
    float d2 = d * sc;
    float m2 = dmin * m;
    
    printf("d1=%.10f m1=%.10f d2=%.10f m2=%.10f\n", d1, m1, d2, m2);
    
    // Dump first 8 dequantized values
    float out[256];
    for (int l = 0; l < 32; l++) {
        out[l] = d1 * (qs[l] & 0xF) - m1;
        out[32+l] = d2 * (qs[l] >> 4) - m2;
    }
    
    printf("First 8 dequantized values: ");
    for (int i = 0; i < 8; i++) printf("%.10f ", out[i]);
    printf("\n");
    printf("Mean of first 256: %.10f\n", (d1*7.5f - m1 + d2*7.5f - m2)/2.0f);
    
    // Save dequantized block
    FILE *f = fopen("/tmp/our_q4k_block.bin", "wb");
    fwrite(out, sizeof(float), 256, f);
    fclose(f);
    
    // Also save raw block for Python verification
    f = fopen("/tmp/raw_q4k_block.bin", "wb");
    fwrite(data, 1, 144, f);
    fclose(f);
    
    printf("Saved dequantized block and raw block\n");
    
    gguf_close(ctx);
    return 0;
}
