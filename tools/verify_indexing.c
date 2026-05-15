/* verify_indexing.c — Check if weight indexing is correct */
#include "gguf_reader.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <math.h>

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    gguf_buffer_data(ctx);
    
    // Check attn_qkv.weight — dims [2048, 8192]
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.attn_qkv.weight");
    printf("attn_qkv.weight: dims=[%ld,%ld] type=%d\n", 
           (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    printf("  GGUF stores element (d0,d1) at index d0 + d1*%ld\n", (long)t->dims[0]);
    
    float *buf = (float *)malloc(100 * sizeof(float));
    gguf_read_tensor_f32(ctx, t, buf, 100);
    
    // The code does: weight[i * 8192 + j] for i=0..2047, j=0..8191
    // GGUF stores W[i][j] (i=row, j=col) at: i + j * 2048
    // Code reads: weight[i * 8192 + j]
    // For first few elements: i=0, j=0..3 and i=1, j=0..3
    printf("\nCode indexing (i*8192+j):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            int idx = i * 8192 + j;
            printf("  i=%d j=%d -> idx=%d val=%.6f\n", i, j, idx, 
                   idx < 100 ? buf[idx] : -999.0f);
        }
    }
    printf("\nGGUF storage (i + j*2048):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            int idx = i + j * 2048;
            printf("  i=%d j=%d -> idx=%d val=%.6f\n", i, j, idx,
                   idx < 100 ? buf[idx] : -999.0f);
        }
    }
    
    // Compare: output.weight — dims [2048, 248320]
    t = gguf_find_tensor(ctx, "output.weight");
    printf("\noutput.weight: dims=[%ld,%ld] type=%d\n",
           (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    
    float *obuf = (float *)malloc(100 * sizeof(float));
    gguf_read_tensor_f32(ctx, t, obuf, 100);
    
    printf("Code indexing (j*2048+k):\n");
    for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 4; k++) {
            int idx = j * 2048 + k;
            printf("  j=%d k=%d -> idx=%d val=%.6f\n", j, k, idx,
                   idx < 100 ? obuf[idx] : -999.0f);
        }
    }
    printf("GGUF storage (k + j*2048):\n");
    for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 4; k++) {
            int idx = k + j * 2048;
            printf("  j=%d k=%d -> idx=%d val=%.6f\n", j, k, idx,
                   idx < 100 ? obuf[idx] : -999.0f);
        }
    }
    
    free(buf); free(obuf);
    gguf_close(ctx);
    return 0;
}
