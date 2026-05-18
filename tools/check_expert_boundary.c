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
    
    printf("Type: %d\n", t->ggml_type);
    printf("Dims: ");
    for(int i=0;i<t->n_dims;i++) printf("%lld ", (long long)t->dims[i]);
    printf("\n");
    
    int64_t n_all = (int64_t)D_MODEL * D_FF * N_EXPERTS;
    int64_t n_one = (int64_t)D_MODEL * D_FF;
    
    int64_t raw_all = gguf_raw_size(t->ggml_type, n_all);
    int64_t raw_one = gguf_raw_size(t->ggml_type, n_one);
    
    printf("n_all=%lld n_one=%lld\n", (long long)n_all, (long long)n_one);
    printf("raw_all=%lld raw_one=%lld\n", (long long)raw_all, (long long)raw_one);
    printf("raw_all/256=%lld\n", (long long)(raw_all/N_EXPERTS));
    printf("Match: %s\n", raw_all == N_EXPERTS * raw_one ? "YES" : "NO");
    
    float *buf = (float*)malloc(n_one * sizeof(float));
    
    for(int e=0; e<4; e++) {
        int64_t off = e * raw_one;
        gguf_dequantize((const uint8_t*)ctx->data_blob + t->data_offset + off,
                        t->ggml_type, n_one, buf);
        float mn=1e30,mx=-1e30;
        for(int64_t i=0;i<n_one;i++){
            if(buf[i]<mn)mn=buf[i];
            if(buf[i]>mx)mx=buf[i];
        }
        printf("Expert %d (off=%lld): min=%.2e max=%.2e first=%.6f\n",
               e, (long long)off, mn, mx, buf[0]);
    }
    
    free(buf);
    gguf_close(ctx);
    return 0;
}
