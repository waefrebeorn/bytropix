#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    
    int64_t total = t->dims[0] * t->dims[1];
    float *buf = (float*)malloc(total * sizeof(float));
    int n = gguf_read_tensor_f32(ctx, t, buf, total);
    printf("Read %d elements\n", n);
    
    // Distribution of first 256 elements (block 0)
    double mean, min, max, median;
    float vals[256];
    for(int i=0;i<256;i++) vals[i]=buf[i];
    mean=0; min=1e30; max=-1e30;
    for(int i=0;i<256;i++){mean+=vals[i];if(vals[i]<min)min=vals[i];if(vals[i]>max)max=vals[i];}
    mean/=256;
    // Sort for median
    for(int i=0;i<256;i++)for(int j=i+1;j<256;j++)if(vals[j]<vals[i]){float t=vals[i];vals[i]=vals[j];vals[j]=t;}
    printf("Block 0: mean=%.4f min=%.4f max=%.4f median=%.4f\n", mean, min, max, vals[128]);
    printf("  P10=%.4f P25=%.4f P75=%.4f P90=%.4f\n", vals[25], vals[64], vals[192], vals[230]);
    printf("  First 16: ");
    for(int i=0;i<16;i++) printf(" %+.4f", buf[i]);
    printf("\n");
    
    // Block 1 (elements 256-511)
    for(int i=0;i<256;i++) vals[i]=buf[256+i];
    mean=0; min=1e30; max=-1e30;
    for(int i=0;i<256;i++){mean+=vals[i];if(vals[i]<min)min=vals[i];if(vals[i]>max)max=vals[i];}
    mean/=256;
    printf("Block 1: mean=%.4f min=%.4f max=%.4f\n", mean, min, max);
    
    // Block 4 (elements 1024-1279)
    for(int i=0;i<256;i++) vals[i]=buf[1024+i];
    mean=0; min=1e30; max=-1e30;
    for(int i=0;i<256;i++){mean+=vals[i];if(vals[i]<min)min=vals[i];if(vals[i]>max)max=vals[i];}
    mean/=256;
    printf("Block 4: mean=%.4f min=%.4f max=%.4f\n", mean, min, max);
    
    // Check all elements for extreme values
    double all_mean=0, all_min=1e30, all_max=-1e30;
    long long extreme_neg=0, extreme_pos=0;
    for(int i=0;i<n;i++){
        all_mean+=buf[i];
        if(buf[i]<all_min)all_min=buf[i];
        if(buf[i]>all_max)all_max=buf[i];
        if(buf[i] < -10000) extreme_neg++;
        if(buf[i] > 10000) extreme_pos++;
    }
    all_mean/=n;
    printf("\nALL %d elems: mean=%.2f min=%.2f max=%.2f\n", n, all_mean, all_min, all_max);
    printf("  <-10K: %lld  >+10K: %lld (out of %d)\n", extreme_neg, extreme_pos, n);
    
    free(buf);
    gguf_close(ctx);
    return 0;
}
