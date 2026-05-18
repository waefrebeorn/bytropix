#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if(!ctx) return 1;
    gguf_buffer_data(ctx);
    
    char namebuf[256];
    snprintf(namebuf,sizeof(namebuf),"blk.0.ffn_gate_exps.weight");
    gguf_tensor_info *tstruct = gguf_find_tensor(ctx,namebuf);
    if(!tstruct) {fprintf(stderr,"not found\n"); return 1;}
    
    printf("Tensor info:\n");
    printf("  name: %s\n", tstruct->name);
    printf("  dims: %d (", tstruct->n_dims);
    for(int i=0;i<tstruct->n_dims;i++) printf("%lld ", (long long)tstruct->dims[i]);
    printf(")\n");
    printf("  ggml_type: %d\n", tstruct->ggml_type);
    printf("  data_offset: %lld\n", (long long)tstruct->data_offset);
    
    int64_t n_elements = (int64_t)D_MODEL * D_FF * N_EXPERTS;
    int64_t raw_total = gguf_raw_size(tstruct->ggml_type, n_elements);
    int64_t expert_raw_bytes = raw_total / N_EXPERTS;
    int64_t expert_elements = (int64_t)D_MODEL * D_FF;
    
    printf("  n_elements (all experts): %lld\n", (long long)n_elements);
    printf("  raw_total: %lld\n", (long long)raw_total);
    printf("  expert_raw (per expert): %lld bytes\n", (long long)expert_raw_bytes);
    printf("  expert_elements (floats): %lld\n", (long long)expert_elements);
    printf("  raw per expert: %lld\n", (long long)gguf_raw_size(tstruct->ggml_type, expert_elements));
    
    // Dequantize expert 0
    float *ex0 = (float*)malloc(expert_elements * sizeof(float));
    gguf_dequantize((const uint8_t*)ctx->data_blob + tstruct->data_offset + 0 * expert_raw_bytes,
        tstruct->ggml_type, expert_elements, ex0);
    
    float *ex1 = (float*)malloc(expert_elements * sizeof(float));
    gguf_dequantize((const uint8_t*)ctx->data_blob + tstruct->data_offset + 1 * expert_raw_bytes,
        tstruct->ggml_type, expert_elements, ex1);
    
    printf("\nExpert 0 vs Expert 1 first 10:\n");
    for(int i=0;i<10;i++){
        printf("  [%d]: ex0=%.6f  ex1=%.6f  diff=%.6f\n", i, ex0[i], ex1[i], ex1[i]-ex0[i]);
    }
    
    float mx=-1e30, minv=1e30;
    for(int64_t i=0;i<expert_elements;i++){
        if(ex0[i]>mx)mx=ex0[i];
        if(ex0[i]<minv)minv=ex0[i];
    }
    printf("\nExpert 0: min=%.6f max=%.6f\n", minv, mx);
    
    printf("Expert 0 first 5: ");
    for(int i=0;i<5;i++) printf("%.6f ", ex0[i]);
    printf("\n");
    
    printf("\nSample gate_out (expert 0, x=1):\n");
    for(int j=0;j<5;j++){
        double sum=0;
        for(int k=0;k<D_MODEL;k++) sum += ex0[k * D_FF + j];
        printf("  gate[%d] = %.2f\n", j, sum);
    }
    
    float *gate_buf = (float*)malloc(D_FF * sizeof(float));
    for(int j=0;j<D_FF;j++){
        double sum=0;
        for(int k=0;k<D_MODEL;k++) sum += 1.0 * ex0[k * D_FF + j];
        gate_buf[j] = (float)sum;
    }
    
    float gmx=-1e30; int gmi=-1;
    for(int j=0;j<D_FF;j++){
        if(fabsf(gate_buf[j])>gmx){gmx=fabsf(gate_buf[j]);gmi=j;}
    }
    printf("\nMax |gate|: %.2f at idx %d (out of %d)\n", gmx, gmi, D_FF);
    printf("Gate first 10: ");
    for(int j=0;j<10;j++) printf("%.2f ", gate_buf[j]);
    printf("\n");
    
    free(ex0); free(ex1); free(gate_buf);
    gguf_close(ctx);
    return 0;
}
