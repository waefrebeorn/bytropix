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
    
    // Check all 3 expert weight tensors for layer 0
    const char *names[] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
    for(int i=0;i<3;i++){
        snprintf(mn,sizeof(mn),"blk.0.%s",names[i]);
        gguf_tensor_info *t = gguf_find_tensor(ctx,mn);
        if(!t) continue;
        printf("%s: type=%d dims=", t->name, t->ggml_type);
        for(int d=0;d<t->n_dims;d++) printf("%lld ", (long long)t->dims[d]);
        printf("\n");
        // Reverse to get Fortran-order shape
        printf("  Fortran shape: [");
        for(int d=t->n_dims-1;d>=0;d--) printf("%lld%s", (long long)t->dims[d], d>0?",":"");
        printf("]\n");
        int64_t total = 1;
        for(int d=0;d<t->n_dims;d++) total *= t->dims[d];
        printf("  total elements: %lld\n", (long long)total);
    }
    
    // Also check gate_inp
    snprintf(mn,sizeof(mn),"blk.0.ffn_gate_inp.weight");
    gguf_tensor_info *t = gguf_find_tensor(ctx,mn);
    if(t){
        printf("\n%s: type=%d dims=", t->name, t->ggml_type);
        for(int d=0;d<t->n_dims;d++) printf("%lld ", (long long)t->dims[d]);
        printf("\n");
        printf("  Fortran shape: [");
        for(int d=t->n_dims-1;d>=0;d--) printf("%lld%s", (long long)t->dims[d], d>0?",":"");
        printf("]\n");
    }
    
    gguf_close(ctx);
    return 0;
}
