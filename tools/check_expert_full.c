#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if(!ctx) return 1;
    gguf_buffer_data(ctx);
    
    int64_t expert_elements = (int64_t)D_MODEL * D_FF; // 2048 * 512 = 1048576
    
    char namebuf[256];
    snprintf(namebuf,sizeof(namebuf),"blk.0.ffn_gate_exps.weight");
    gguf_tensor_info *t = gguf_find_tensor(ctx, namebuf);
    if(!t) return 1;
    
    int64_t expert_raw = gguf_raw_size(t->ggml_type, (int64_t)D_MODEL*D_FF*N_EXPERTS) / N_EXPERTS;
    printf("Expert raw bytes: %lld\n", (long long)expert_raw);
    
    float *ex = (float*)malloc(expert_elements * sizeof(float));
    gguf_dequantize((const uint8_t*)ctx->data_blob + t->data_offset, t->ggml_type,
                    expert_elements, ex);
    
    printf("Expert 0 full scan:\n");
    float mn=1e30,mx=-1e30;
    int mni=-1,mxi=-1;
    for(int64_t i=0;i<expert_elements;i++){
        if(ex[i]<mn){mn=ex[i];mni=i;}
        if(ex[i]>mx){mx=ex[i];mxi=i;}
    }
    printf("  min=%.6e idx=%d max=%.6e idx=%d\n", mn, mni, mx, mxi);
    
    printf("Expert 0 sparse samples:\n");
    for(int64_t i=0;i<expert_elements;i+=expert_elements/10){
        printf("  [%lld]: %.6f\n", (long long)i, ex[i]);
    }
    printf("\n");
    
    // Now check expert 1
    float *ex1 = (float*)malloc(expert_elements * sizeof(float));
    gguf_dequantize((const uint8_t*)ctx->data_blob + t->data_offset + expert_raw,
                    t->ggml_type, expert_elements, ex1);
    
    printf("Expert 1 full scan:\n");
    mn=1e30;mx=-1e30;mni=-1;mxi=-1;
    for(int64_t i=0;i<expert_elements;i++){
        if(ex1[i]<mn){mn=ex1[i];mni=i;}
        if(ex1[i]>mx){mx=ex1[i];mxi=i;}
    }
    printf("  min=%.6e idx=%d max=%.6e idx=%d\n", mn, mni, mx, mxi);
    
    printf("Expert 1 sparse samples:\n");
    for(int64_t i=0;i<expert_elements;i+=expert_elements/10){
        printf("  [%lld]: %.6f\n", (long long)i, ex1[i]);
    }
    
    // Simulate full expert forward with x=1 for expert 0
    printf("\nSimulating expert 0 forward with x=1:\n");
    float *gate = (float*)calloc(D_FF, sizeof(float));
    float *up = (float*)calloc(D_FF, sizeof(float));
    float *act = (float*)calloc(D_FF, sizeof(float));
    
    // gate[j] = sum_k x[k] * w[k * D_FF + j]
    for(int j=0;j<D_FF;j++){
        double sum=0;
        for(int k=0;k<D_MODEL;k++) sum += 1.0 * ex[k * D_FF + j];
        gate[j] = (float)sum;
    }
    
    float gmin=1e30,gmax=-1e30;
    for(int j=0;j<D_FF;j++){
        if(gate[j]<gmin)gmin=gate[j];
        if(gate[j]>gmax)gmax=gate[j];
    }
    printf("  gate: min=%.2f max=%.2f\n", gmin, gmax);
    printf("  gate first 5: ");
    for(int j=0;j<5;j++) printf("%.2f ", gate[j]);
    printf("\n");
    
    // Now test with x from actual model
    // Print actual FFN intermediates
    free(ex); free(ex1); free(gate); free(up); free(act);
    gguf_close(ctx);
    return 0;
}
