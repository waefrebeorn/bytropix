#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>

// Check GGUF dim order: dims[0] innermost or outermost?
// If conv1d.weight has dims=[C,k], and data[0..C-1] = filter0's channel weights,
// then dims[0]=C is innermost -> old code `ki*C+c` is correct, new `ki+c*k` is wrong.
// If dims[0]=k is innermost -> new code correct.

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { fprintf(stderr, "FAIL: open\n"); return 1; }
    
    const char *tnames[] = {
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
        "blk.0.post_attention_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_dt.bias",
        "blk.0.ssm_a",
        "blk.0.ssm_conv1d.weight",
        "blk.0.ssm_norm.weight",
        "blk.0.ssm_out.weight",
        "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_up_shexp.weight",
        "blk.0.ffn_down_shexp.weight",
        "output_norm.weight",
        "output.weight",
        NULL
    };
    
    printf("=== GGUF Tensor Dimensions ===\n");
    printf("GGUF spec: dims[0] = innermost (fastest-varying in memory)\n");
    printf("For 2D weight matrix shape (output_dim, input_dim):\n");
    printf("  dims[0] = input_dim (innermost)\n");
    printf("  dims[1] = output_dim (outermost)\n");
    printf("Index for W[output][input]: input + output * dims[0]\n");
    printf("\n");
    
    for (int k = 0; tnames[k]; k++) {
        gguf_tensor_info *t = gguf_find_tensor(ctx, tnames[k]);
        if (!t) {
            printf("  %-32s NOT FOUND\n", tnames[k]);
            continue;
        }
        printf("  %-32s dims=[%ld", tnames[k], t->dims[0]);
        for (int d = 1; d < t->n_dims; d++) printf(", %ld", t->dims[d]);
        printf("] n_dims=%d\n", t->n_dims);
    }
    
    // Deep verify: read conv1d values and check indexing
    gguf_tensor_info *conv1d = gguf_find_tensor(ctx, "blk.0.ssm_conv1d.weight");
    if (conv1d) {
        printf("\n=== ssm_conv1d.weight: dims=[%ld,%ld]\n", conv1d->dims[0], conv1d->dims[1]);
        printf("dim0=%ld (innermost), dim1=%ld (outermost)\n", conv1d->dims[0], conv1d->dims[1]);
        printf("For kernel[ki][c] where ki=0..k-1, c=0..C-1:\n");
        printf("  Old code: ki*C + c = ki*%ld + c\n", conv1d->dims[0]);
        printf("  New code: ki + c*k = ki + c*%ld\n", conv1d->dims[1]);
        printf("  GGUF correct: c + ki*%ld (inner + outer*inner_dim)\n", conv1d->dims[0]);
        printf("  Old matches GGUF? %s\n", "YES if dims[0]==C (innermost=channel)");
        printf("  New matches GGUF? %s\n", "YES if dims[0]==k (innermost=kernel)");
        
        // Read a few values
        int64_t nelem = conv1d->dims[0] * conv1d->dims[1];
        float *buf = (float*)malloc(nelem * sizeof(float));
        int got = gguf_read_tensor_f32(ctx, conv1d, buf, nelem);
        if (got > 0) {
            printf("\n  First 12 values: ");
            for (int i = 0; i < 12; i++) printf("%.6f ", buf[i]);
            printf("\n");
            // Values by old index: ki*C+c
            printf("  Old idx(ki=0,c=0..3): ");
            for (int c = 0; c < 4; c++) printf("%.6f ", buf[0*conv1d->dims[0] + c]);
            printf("\n  Old idx(ki=1,c=0..3): ");
            for (int c = 0; c < 4; c++) printf("%.6f ", buf[1*conv1d->dims[0] + c]);
            printf("\n");
            // Values by new index: ki + c*k
            printf("  New idx(ki=0,c=0..3): ");
            for (int c = 0; c < 4; c++) printf("%.6f ", buf[0 + c*conv1d->dims[1]]);
            printf("\n  New idx(ki=1,c=0..3): ");
            for (int c = 0; c < 4; c++) printf("%.6f ", buf[1 + c*conv1d->dims[1]]);
            printf("\n");
            
            // Which pattern shows variation?
            printf("\n  Pattern check: First 4 values should be filter0's first 4 channels\n");
            printf("  If filter0 channels vary -> dims[0]=%ld=CONV_DIM is innermost (old code correct)\n", conv1d->dims[0]);
            printf("  If filter0 channels constant -> dims[0]=%ld=kernel is innermost (new code correct)\n", conv1d->dims[1]);
        }
        free(buf);
    }
    
    // Same for attn_qkv
    gguf_tensor_info *qkv = gguf_find_tensor(ctx, "blk.0.attn_qkv.weight");
    if (!qkv) qkv = gguf_find_tensor(ctx, "blk.0.ssm.attn_qkv.weight");
    if (qkv) {
        printf("\n=== attn_qkv.weight: dims=[%ld,%ld]\n", qkv->dims[0], qkv->dims[1]);
        printf("  Shape (output_dim, input_dim) = (%ld, %ld)\n", qkv->dims[1], qkv->dims[0]);
        printf("  Index for W[j][i] (j=output, i=input): i + j*%ld\n", qkv->dims[0]);
        printf("  Old code: i*%ld + j -> %s\n", qkv->dims[1],
               (qkv->dims[0] == 2048) ? "WRONG (row-major)" : "depends");
        printf("  New code: i + j*%ld -> %s\n", qkv->dims[0],
               (qkv->dims[0] == 2048) ? "CORRECT" : "check");
    }
    
    gguf_close(ctx);
    return 0;
}
