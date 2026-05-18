/* Scan model weights for NaN/Inf values that could corrupt output */
#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <math.h>

void scan(const char *name, float *data, int64_t n, int max_print) {
    int nan_count = 0, inf_count = 0, zero_count = 0, strange = 0;
    for (int64_t i = 0; i < n; i++) {
        if (isnan(data[i])) nan_count++;
        else if (isinf(data[i])) inf_count++;
        else if (data[i] == 0.0f) zero_count++;
        else if (fabsf(data[i]) > 100.0f) { strange++; if (strange <= max_print) printf("  %s huge: [%lld]=%e\n", name, (long long)i, data[i]); }
    }
    printf("  %s: %lld elems, nan=%d inf=%d zeros=%d huge=%d\n", 
           name, (long long)n, nan_count, inf_count, zero_count, strange);
}

int main() {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    
    printf("Scanning weights:\n");
    scan("attn_norm[0]", mdl.layers[0].attn_norm_weight, D_MODEL, 3);
    scan("attn_qkv[0]", mdl.layers[0].ssm.attn_qkv_weight, D_MODEL * CONV_DIM, 3);
    scan("attn_gate[0]", mdl.layers[0].ssm.attn_gate_weight, D_MODEL * VALUE_DIM, 3);
    scan("ssm_beta[0]", mdl.layers[0].ssm.ssm_beta_weight, D_MODEL * DT_RANK, 3);
    scan("ssm_alpha[0]", mdl.layers[0].ssm.ssm_alpha_weight, D_MODEL * DT_RANK, 3);
    scan("ssm_conv1d[0]", mdl.layers[0].ssm.ssm_conv1d_weight, CONV_KERNEL * CONV_DIM, 3);
    scan("ssm_norm[0]", mdl.layers[0].ssm.ssm_norm_weight, SSM_D_STATE, 3);
    scan("ssm_out[0]", mdl.layers[0].ssm.ssm_out_weight, VALUE_DIM * D_MODEL, 3);
    scan("ssm_dt[0]", mdl.layers[0].ssm.ssm_dt_bias, DT_RANK, 3);
    scan("ssm_a[0]", mdl.layers[0].ssm.ssm_a, DT_RANK, 3);
    scan("output_w", mdl.output_weight, (int64_t)D_MODEL * 248320, 5);
    
    wubu_model_free(&mdl);
    return 0;
}
