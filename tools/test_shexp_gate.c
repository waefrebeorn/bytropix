#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <math.h>

int main() {
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { fprintf(stderr, "Failed to open GGUF\n"); return 1; }
    
    moe_weights_t moe;
    memset(&moe, 0, sizeof(moe));
    
    if (!wubu_moe_load_layer(ctx, 0, &moe)) {
        fprintf(stderr, "Failed to load MoE layer 0\n");
        gguf_close(ctx);
        return 1;
    }
    
    printf("ffn_gate_inp_shexp: %s\n", moe.ffn_gate_inp_shexp ? "LOADED" : "NULL");
    if (moe.ffn_gate_inp_shexp) {
        float sum = 0, min = 1e30, max = -1e30;
        for (int i = 0; i < D_MODEL; i++) {
            float v = moe.ffn_gate_inp_shexp[i];
            sum += v;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        printf("  stats: mean=%.6f min=%.6f max=%.6f\n", sum/D_MODEL, min, max);
        
        // Test: compute gate for a zero input (should give sigmoid(0) = 0.5)
        float input[2048] = {0};  // all zeros
        float dot = 0;
        for (int k = 0; k < D_MODEL; k++) dot += input[k] * moe.ffn_gate_inp_shexp[k];
        float gate = 1.0f / (1.0f + expf(-dot));
        printf("  gate(zero input) = %.6f, sigmoid(%.6f)\n", gate, dot);
        
        // Test: compute gate for all-1 input
        float dot2 = 0;
        for (int k = 0; k < D_MODEL; k++) dot2 += 1.0f * moe.ffn_gate_inp_shexp[k];
        float gate2 = 1.0f / (1.0f + expf(-dot2));
        printf("  gate(all-1 input) = %.6f, sigmoid(%.6f)\n", gate2, dot2);
    }
    
    wubu_moe_free_layer(&moe);
    gguf_close(ctx);
    return 0;
}
