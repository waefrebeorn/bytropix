/**
 * test_ssm_precision.c
 * Feed the reference's L0 hidden state into our SSM at L1.
 * If cos-sim improves dramatically, the bug is precision propagation.
 * If cos-sim stays bad, there's a real bug in our SSM.
 */
#include "gguf_reader.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define D_MODEL  2048
#define LAYER    1  // Test L1

int main() {
    printf("=== SSM Precision Test at Layer %d ===\n", LAYER);
    
    // Load model
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) return 1;
    
    // Load SSM weights for layer 1  
    ssm_layer_weights w;
    if (!wubu_load_ssm_layer(ctx, LAYER, &w)) { printf("FAIL: load SSM L%d\n", LAYER); return 1; }
    
    // Load our L0 output (the full hidden state we store)
    float *h0_our = (float *)malloc(D_MODEL * sizeof(float));
    // We need the FULL hidden state (after residual), not just MoE output
    // Reconstruct: we don't have this stored!
    // Let me check what's saved...
    
    // Alternative: load the BOS embedding (same for both)
    float *embd = (float *)malloc(D_MODEL * sizeof(float));
    FILE *f = fopen("/home/wubu/bytropix/data/qwen36_embeddings_c.bin.raw", "rb");
    fseek(f, 248044LL * D_MODEL * sizeof(float), SEEK_SET);
    fread(embd, sizeof(float), D_MODEL, f);
    fclose(f);
    
    // Load our MoE output at L0
    float *moe0_our = (float *)malloc(D_MODEL * sizeof(float));
    f = fopen("/tmp/dump_layers/our_layer_0_out.bin", "rb");
    fread(moe0_our, sizeof(float), D_MODEL, f);
    fclose(f);
    
    // Reconstruct our hidden state after L0: h = embd + moe0 + shared_expert_output
    // But we don't have shared expert output saved. We have the full hidden state
    // from the model... actually test_full_moe doesn't save the full state.
    
    // Let me instead run wubu_model_forward for one layer and capture the post-attention norm
    // Actually simpler: just compute our post-attention norm at L1 from scratch
    
    // For now: verify we can load and run the SSM
    float *x = (float *)malloc(D_MODEL * sizeof(float));
    float *out = (float *)malloc(D_MODEL * sizeof(float));
    
    // Use embedding as test input
    memcpy(x, embd, D_MODEL * sizeof(float));
    
    // Process through SSM
    wubu_ssm_forward(x, 1, 1, &w, out);
    
    printf("Input embd: mean=%.6f std=%.6f\n", 
           mean_std(embd, D_MODEL).first, mean_std(embd, D_MODEL).second);
    printf("SSM output: mean=%.6f std=%.6f\n",
           mean_std(out, D_MODEL).first, mean_std(out, D_MODEL).second);
    
    // Compare our SSM at L0 against reference
    float *ref_l0_full = (float *)malloc(D_MODEL * sizeof(float));
    f = fopen("/tmp/dump_layers/ref_layer_0.bin", "rb");  // MoE output at L0
    if (f) {
        fread(ref_l0_full, sizeof(float), D_MODEL, f);
        fclose(f);
        double dot=0, na=0, nb=0;
        for (int i = 0; i < D_MODEL; i++) {
            dot += (double)moe0_our[i] * (double)ref_l0_full[i];
            na += (double)moe0_our[i] * (double)moe0_our[i];
            nb += (double)ref_l0_full[i] * (double)ref_l0_full[i];
        }
        printf("\nL0 MoE output vs ref: cos=%.10f std_our=%.6f std_ref=%.6f\n",
               dot/(sqrt(na)*sqrt(nb)), sqrt(na/D_MODEL), sqrt(nb/D_MODEL));
    }
    
    free(embd); free(moe0_our); free(x); free(out); free(ref_l0_full);
    wubu_free_ssm_layer(&w);
    gguf_close(ctx);
    return 0;
}
