#include "gguf_reader.h"
#include <stdio.h>
int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    printf("output.weight: type=%d (C enum)\n", t->ggml_type);
    printf("  GGML_TYPE_Q4_K=%d, GGML_TYPE_Q5_K=%d, GGML_TYPE_Q6_K=%d, GGML_TYPE_Q8_K=%d\n",
           GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_K);
    printf("  GGML_TYPE_IQ2_XXS=%d, GGML_TYPE_IQ2_XS=%d, GGML_TYPE_IQ2_S=%d, GGML_TYPE_IQ1_S=%d\n",
           GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ1_S);
    
    // Read the first 256 elements and dump stats
    float buf[256];
    int n = gguf_read_tensor_f32(ctx, t, buf, 256);
    double mean=0, min=1e30, max=-1e30;
    for(int i=0;i<256;i++){mean+=buf[i];if(buf[i]<min)min=buf[i];if(buf[i]>max)max=buf[i];}
    mean/=256;
    printf("First 256 elems: mean=%.4f min=%.4f max=%.4f\n", mean, min, max);
    printf("First 8 elems: ");
    for(int i=0;i<8;i++) printf("%.4f ", buf[i]);
    printf("\n");
    
    // Read another block far into the tensor (output_weight[0, 50000] or similar)
    // Actually the weight is [2048, 248320] in GGUF = [IN_DIM=2048, OUT_DIM=248320]
    // Stored as 2048 * 248320 = 508M elements
    // We read from offset 1000 * 256 = 256000
    float *big = (float*)malloc(2048 * sizeof(float));
    n = gguf_read_tensor_f32(ctx, t, big, 2048 * 248321);  // full read
    printf("Full read: %d elements / %ld expected\n", n, 2048L * 248320L);
    
    // Check row 0 (first 248320 elements)
    double r0_mean=0, r0_min=1e30, r0_max=-1e30;
    for(int i=0;i<248320;i++){r0_mean+=big[i];if(big[i]<r0_min)r0_min=big[i];if(big[i]>r0_max)r0_max=big[i];}
    r0_mean/=248320;
    printf("Row 0 (248320 elems): mean=%.4f min=%.4f max=%.4f\n", r0_mean, r0_min, r0_max);
    
    // Check row 1
    double r1_mean=0;
    for(int i=248320;i<2*248320;i++){r1_mean+=big[i];}
    r1_mean/=248320;
    printf("Row 1 mean=%.4f\n", r1_mean);
    
    free(big);
    gguf_close(ctx);
    return 0;
}
