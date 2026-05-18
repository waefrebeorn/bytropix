#include "gguf_reader.h"
#include <stdio.h>

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.ssm_out.weight");
    if (!t) { printf("NOT FOUND\n"); return 1; }
    printf("ssm_out.weight: dims=[%ld,%ld] n_dims=%d type=%d\n", 
           (long)t->dims[0], (long)t->dims[1], t->n_dims, t->ggml_type);
    
    int64_t n_elems = t->dims[0] * t->dims[1];
    float *w = malloc(n_elems * sizeof(float));
    gguf_read_tensor_f32(ctx, t, w, n_elems);
    
    printf("First 10: ");
    for (int i = 0; i < 10; i++) printf("%.6f ", w[i]);
    printf("\n");
    printf("w[0]=%.6f  w[4096]=%.6f (D=2048 stride=VALUE_DIM=4096)\n", w[0], w[4096]);
    printf("w[2047]=%.6f w[2048]=%.6f\n", w[2047], w[2048]);
    
    // Also attn_qkv
    t = gguf_find_tensor(ctx, "blk.0.attn_qkv.weight");
    if (t) {
        n_elems = t->dims[0] * t->dims[1];
        float *qkv = malloc(n_elems * sizeof(float));
        gguf_read_tensor_f32(ctx, t, qkv, n_elems);
        printf("attn_qkv dims=[%ld,%ld] type=%d\n", (long)t->dims[0], (long)t->dims[1], t->ggml_type);
        printf("First 5: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", qkv[i]);
        printf("\n");
        printf("w[0+0*2048]=%.6f  w[0+1*2048]=%.6f\n", qkv[0], qkv[0+1*2048]);
        free(qkv);
    }
    
    free(w);
    gguf_close(ctx);
    return 0;
}
