/* verify_indexing2.c — Direct raw data access */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    gguf_buffer_data(ctx);
    
    // Check attn_qkv.weight — dims [2048, 8192]
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.attn_qkv.weight");
    printf("attn_qkv.weight: dims=[%ld,%ld] type=%d\n", 
           (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    printf("  GGUF stores element (d0,d1) at index d0 + d1*%ld\n", (long)t->dims[0]);
    printf("  Code expects element (i,j) at index i*8192 + j\n");
    
    // Dequant first 4096 elements (2 rows worth)
    float *buf = (float *)malloc(4096 * sizeof(float));
    const uint8_t *raw = (const uint8_t *)ctx->data_blob + t->data_offset;
    gguf_dequantize(raw, t->ggml_type, 4096, buf);
    
    printf("\nFirst 8 elements in GGUF order (d0, d1):\n");
    for (int d0 = 0; d0 < 4; d0++) {
        for (int d1 = 0; d1 < 4; d1++) {
            int idx = d0 + d1 * 2048;
            printf("  (d0=%d,d1=%d) -> idx=%d val=%.6f\n", d0, d1, idx, buf[idx]);
        }
    }
    
    // What the code accesses for i=0, j=0..3
    printf("\nCode access: weight[i*8192+j] for i=0, j=0..3:\n");
    for (int j = 0; j < 4; j++) {
        int idx = 0 * 8192 + j;  // i=0
        printf("  i=0 j=%d -> idx=%d val=%.6f (GGUF d0=%d,d1=%d)\n", j, idx, buf[idx], j, 0);
    }
    
    // What the code accesses for i=1, j=0..3
    printf("\nCode access: weight[i*8192+j]:\n");
    for (int j = 0; j < 4; j++) {
        int idx = 1 * 8192 + j;  // i=1
        printf("  i=1 j=%d -> idx=%d val=%.6f (GGUF d0=%d,d1=%d)\n", j, idx, 
               idx < 4096 ? buf[idx] : -999.0f, 
               idx % 2048, idx / 2048);
    }
    
    // What the CORRECT access should be: i + j*2048
    printf("\nCORRECT access: weight[i + j*2048]:\n");
    for (int j = 0; j < 4; j++) {
        int idx = 1 + j * 2048;  // i=1
        printf("  i=1 j=%d -> idx=%d val=%.6f\n", j, idx, buf[idx]);
    }
    
    // Compare stats: first row vs "row 1" via code indexing
    double code_row1[4];
    double correct_row1[4];
    for (int j = 0; j < 4; j++) {
        code_row1[j] = buf[1 * 8192 + j];  // Wrong!
        correct_row1[j] = buf[1 + j * 2048];  // Correct!
    }
    printf("\nRow i=1, code gets: [%.4f, %.4f, %.4f, %.4f]\n",
           code_row1[0], code_row1[1], code_row1[2], code_row1[3]);
    printf("Row i=1, GGUF has: [%.4f, %.4f, %.4f, %.4f]\n",
           correct_row1[0], correct_row1[1], correct_row1[2], correct_row1[3]);
    
    // For output.weight  
    t = gguf_find_tensor(ctx, "output.weight");
    printf("\noutput.weight: dims=[%ld,%ld] type=%d\n",
           (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    printf("  Code: output_weight[j*2048+k]\n");
    printf("  GGUF: element (d0=k, d1=j) at k + j*2048\n");
    printf("  These are IDENTICAL since j*2048+k = k+j*2048\n");
    
    free(buf);
    gguf_close(ctx);
    return 0;
}
