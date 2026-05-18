#include "gguf_reader.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) return 1;

    // Check GQA layer 3 weights
    const char *tensors[] = {
        "blk.3.attn_q.weight",
        "blk.3.attn_k.weight", 
        "blk.3.attn_v.weight",
        "blk.3.attn_output.weight",
        "blk.3.attn_q_norm.weight",
        "blk.3.attn_k_norm.weight",
        NULL
    };

    for (int ti = 0; tensors[ti]; ti++) {
        gguf_tensor_info *t = gguf_find_tensor(ctx, tensors[ti]);
        if (!t) { printf("%s: NOT FOUND\n", tensors[ti]); continue; }
        
        int n_elems = 1;
        for (int d = 0; d < t->n_dims; d++) n_elems *= t->dims[d];
        
        // Read first 8 elements
        float *tmp = (float *)malloc(8 * sizeof(float));
        gguf_read_tensor_f32(ctx, t, tmp, 8);
        
        printf("%s: type=%d dims=[", tensors[ti], t->ggml_type);
        for (int d = 0; d < t->n_dims; d++) printf("%ld,", (long)t->dims[d]);
        printf("] first8=");
        int has_nan = 0;
        for (int i = 0; i < 8; i++) {
            printf("%+.4e ", (double)tmp[i]);
            if (isnan(tmp[i])) has_nan = 1;
        }
        printf("  %s\n", has_nan ? "*** NaN in first 8 ***" : "OK");
        free(tmp);
        
        // If it's a weight tensor (not norm), check for NaN in a few more spots
        if (t->n_dims >= 2) {
            float *full = (float *)malloc(n_elems * sizeof(float));
            int nread = gguf_read_tensor_f32(ctx, t, full, n_elems);
            int nan_count = 0, first_nan_idx = -1, first_nan_col = -1;
            for (int i = 0; i < nread; i++) {
                if (isnan(full[i])) {
                    nan_count++;
                    if (first_nan_idx < 0) {
                        first_nan_idx = i;
                        first_nan_col = i % t->dims[1];
                    }
                }
            }
            printf("  NaN count: %d / %d", nan_count, nread);
            if (nan_count > 0) printf(" first at idx=%d col=%d", first_nan_idx, first_nan_col);
            printf("\n");
            free(full);
        }
    }
    
    gguf_close(ctx);
    return 0;
}
