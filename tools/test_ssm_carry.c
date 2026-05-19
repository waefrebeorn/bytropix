/**
 * Test: SSM forward with T=6 + T=1 vs T=7
 * Verifies state carry consistency
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, path)) return 1;
    
    // Get layer 0 SSM weights
    ssm_layer_weights *w = &mdl.layers[0].ssm;
    
    // Create fake input: 7 tokens, D_MODEL random values
    int T_full = 7;
    int T_pre = 6;
    float *x = (float *)malloc(T_full * D_MODEL * sizeof(float));
    for (int i = 0; i < T_full * D_MODEL; i++) x[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    
    // State buffers
    float *ssm_state_full = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
    float *conv_state_full = (float *)calloc((CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
    float *ssm_state_part = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
    float *conv_state_part = (float *)calloc((CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
    
    // Output buffers
    float *out_full = (float *)malloc(T_full * D_MODEL * sizeof(float));
    float *out_part_6 = (float *)malloc(T_pre * D_MODEL * sizeof(float));
    float *out_extra = (float *)malloc(1 * D_MODEL * sizeof(float));
    
    // Run T=7 (full batch)
    wubu_ssm_forward(x, 1, T_full, w, ssm_state_full, conv_state_full, out_full, NULL, NULL);
    
    // Run T=6 (partial batch)
    wubu_ssm_forward(x, 1, T_pre, w, ssm_state_part, conv_state_part, out_part_6, NULL, NULL);
    
    // Run T=1 (one more token) using the state from T=6
    const float *x_extra = x + T_pre * D_MODEL;
    wubu_ssm_forward(x_extra, 1, 1, w, ssm_state_part, conv_state_part, out_extra, NULL, NULL);
    
    // Compare last-token outputs
    printf("=== SSM State Carry Test ===\n");
    printf("T_full=%d, T_pre=%d, T_extra=1\n", T_full, T_pre);
    
    // Compare out_full[T_pre] (6th token from T=7) vs out_extra[0] (1st token from T=1)
    const float *ref = out_full + T_pre * D_MODEL;
    const float *cmp = out_extra;
    
    float max_diff = 0, sum_diff = 0;
    int max_idx = 0;
    for (int i = 0; i < D_MODEL; i++) {
        float d = fabsf(ref[i] - cmp[i]);
        sum_diff += d;
        if (d > max_diff) { max_diff = d; max_idx = i; }
    }
    
    printf("T=7 token 6 vs T=1 extra:\n");
    printf("  max_diff=%.8f (idx=%d)\n", max_diff, max_idx);
    printf("  mean_diff=%.8f\n", sum_diff / D_MODEL);
    printf("  ref[0..4]=%.6f %.6f %.6f %.6f %.6f\n", ref[0], ref[1], ref[2], ref[3], ref[4]);
    printf("  cmp[0..4]=%.6f %.6f %.6f %.6f %.6f\n", cmp[0], cmp[1], cmp[2], cmp[3], cmp[4]);
    
    // Also compare ssm_state
    int state_sz = SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    float ss_max = 0;
    int ss_idx = 0;
    for (int i = 0; i < state_sz; i++) {
        float d = fabsf(ssm_state_full[i] - ssm_state_part[i]);
        if (d > ss_max) { ss_max = d; ss_idx = i; }
    }
    printf("SSM state diff: max=%.8f at %d\n", ss_max, ss_idx);
    
    // Conv state diff
    int conv_sz = (CONV_KERNEL - 1) * CONV_DIM;
    float cs_max = 0;
    int cs_idx = 0;
    for (int i = 0; i < conv_sz; i++) {
        float d = fabsf(conv_state_full[i] - conv_state_part[i]);
        if (d > cs_max) { cs_max = d; cs_idx = i; }
    }
    printf("Conv state diff: max=%.8f at %d\n", cs_max, cs_idx);
    
    wubu_model_free(&mdl);
    return 0;
}
