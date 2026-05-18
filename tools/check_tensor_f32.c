#include "gguf_reader.h"
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if(!ctx) return 1;
    gguf_buffer_data(ctx);
    
    char mn[256];
    snprintf(mn,sizeof(mn),"blk.0.ffn_gate_exps.weight");
    gguf_tensor_info *t = gguf_find_tensor(ctx, mn);
    if(!t) return 1;
    
    printf("Type: %d Name: %s\n", t->ggml_type, t->name);
    printf("n_dims: %d\n", t->n_dims);
    for(int i=0;i<t->n_dims;i++) printf("  dim[%d]=%lld\n", i, (long long)t->dims[i]);
    
    // Read ALL experts as F32
    int64_t n_all = (int64_t)D_MODEL * D_FF * N_EXPERTS;
    float *all = (float*)malloc(n_all * sizeof(float));
    int64_t read = gguf_read_tensor_f32(ctx, t, all, n_all);
    printf("Read %lld floats\n", (long long)read);
    
    int64_t n_one = (int64_t)D_MODEL * D_FF;
    for(int e=0; e<4; e++) {
        float *ex = all + e * n_one;
        float mn=1e30,mx=-1e30;
        for(int64_t i=0;i<n_one;i++){
            if(ex[i]<mn)mn=ex[i];
            if(ex[i]>mx)mx=ex[i];
        }
        printf("Expert %d (F32 read): min=%.2e max=%.2e\n", e, mn, mx);
    }
    
    free(all);
    gguf_close(ctx);
    return 0;
}
