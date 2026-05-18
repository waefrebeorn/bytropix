#include "gguf_reader.h"
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if(!ctx) return 1;
    gguf_buffer_data(ctx);
    
    char mn[256];
    snprintf(mn,sizeof(mn),"blk.0.ffn_gate_exps.weight");
    gguf_tensor_info *t = gguf_find_tensor(ctx, mn);
    if(!t) return 1;
    
    uint8_t *base = (uint8_t*)ctx->data_blob + t->data_offset;
    int64_t raw_one = gguf_raw_size(t->ggml_type, (int64_t)D_MODEL * D_FF);
    
    // Decode the fp16 scale at start of each block for first 10 blocks
    // and for blocks at expert boundaries
    printf("Scales at block starts (fp16 bytes → float):\n");
    for(int b=0; b<20; b++) {
        int64_t off = b * 72;
        uint16_t d_bits;
        memcpy(&d_bits, base + off, 2);
        float d;
        // f16_to_f32 inline
        {
            int s = (d_bits >> 15) & 1;
            int e = (d_bits >> 10) & 31;
            int m = d_bits & 1023;
            if (e == 0) d = (float)(m) / 16777216.0f * (s ? -1.0f : 1.0f);
            else if (e == 31) d = m ? NAN : (s ? -INFINITY : INFINITY);
            else d = (float)((1 << 10) | m) * (s ? -1.0f : 1.0f) / (float)(1 << (15 - e + 127 - 23));
        }
        printf("  block %2d (off=%6lld): d_bits=0x%04x d=%f raw_bytes: %02x %02x\n",
               b, (long long)off, d_bits, d, base[off], base[off+1]);
    }
    
    // Print blocks at expert boundaries (b=0, 4096, 8192)
    printf("\nExpert boundary scales:\n");
    for(int e=0; e<4; e++) {
        int64_t b = e * 4096;
        int64_t off = b * 72;
        uint16_t d_bits;
        memcpy(&d_bits, base + off, 2);
        int s = (d_bits >> 15) & 1;
        int ebits = (d_bits >> 10) & 31;
        int m = d_bits & 1023;
        float d;
        if (ebits == 0) d = (float)(m) / 16777216.0f * (s ? -1.0f : 1.0f);
        else if (ebits == 31) d = m ? NAN : (s ? -INFINITY : INFINITY);
        else d = (float)((1 << 10) | m) * (s ? -1.0f : 1.0f) / (float)(1 << (15 - ebits + 127 - 23));
        printf("  expert %d block %lld (off=%lld): d_bits=0x%04x d=%f raw: %02x %02x\n",
               e, (long long)b, (long long)off, d_bits, d, base[off], base[off+1]);
    }
    
    // Now decode with full gguf_read_tensor_f32 and check specific positions
    int64_t n_all = (int64_t)D_MODEL * D_FF * N_EXPERTS;
    float *all = (float*)malloc(n_all * sizeof(float));
    gguf_read_tensor_f32(ctx, t, all, n_all);
    
    printf("\nFloat values at expert boundaries (first 3 of each):\n");
    for(int e=0; e<4; e++) {
        float *ex = all + e * (int64_t)D_MODEL * D_FF;
        printf("  expert %d: %.6f %.6f %.6f\n", e, ex[0], ex[1], ex[2]);
    }
    
    // Dequant from block 0
    float *buf = (float*)malloc((int64_t)D_MODEL * D_FF * sizeof(float));
    gguf_dequantize(base, t->ggml_type, (int64_t)D_MODEL * D_FF, buf);
    printf("\nDequant from off=0: buf[0]=%f buf[1]=%f\n", buf[0], buf[1]);
    
    // Dequant from 4096 blocks into
    gguf_dequantize(base + 4096 * 72, t->ggml_type, (int64_t)D_MODEL * D_FF, buf);
    printf("Dequant from off=%lld (block 4096): buf[0]=%f buf[1]=%f\n", 
           (long long)(4096 * 72), buf[0], buf[1]);
    
    // Dequant from 1*raw_one
    gguf_dequantize(base + raw_one, t->ggml_type, (int64_t)D_MODEL * D_FF, buf);
    printf("Dequant from off=%lld (1*raw_one): buf[0]=%f buf[1]=%f\n",
           (long long)raw_one, buf[0], buf[1]);
    
    free(all); free(buf);
    gguf_close(ctx);
    return 0;
}
