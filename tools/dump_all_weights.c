/* Dump ALL layer 0 weights as flat float files for Python comparison */
#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>

void dump(const char *label, float *data, int n) {
    char path[256];
    snprintf(path, sizeof(path), "/tmp/c_%s.bin", label);
    FILE *f = fopen(path, "wb");
    fwrite(data, sizeof(float), n, f);
    fclose(f);
    printf("Dumped %s: %d floats\n", label, n);
}

int main() {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    
    // Layer 0 weights
    ssm_layer_weights *w = &mdl.layers[0].ssm;
    dump("attn_norm", mdl.layers[0].attn_norm_weight, D_MODEL);
    dump("post_attn_norm", mdl.layers[0].post_attn_norm_weight, D_MODEL);
    dump("attn_qkv", w->attn_qkv_weight, D_MODEL * CONV_DIM);
    dump("attn_gate", w->attn_gate_weight, D_MODEL * VALUE_DIM);
    dump("ssm_beta", w->ssm_beta_weight, D_MODEL * DT_RANK);
    dump("ssm_alpha", w->ssm_alpha_weight, D_MODEL * DT_RANK);
    dump("ssm_conv1d", w->ssm_conv1d_weight, CONV_KERNEL * CONV_DIM);
    dump("ssm_norm", w->ssm_norm_weight, SSM_D_STATE);
    dump("ssm_out", w->ssm_out_weight, VALUE_DIM * D_MODEL);
    dump("ssm_dt", w->ssm_dt_bias, DT_RANK);
    dump("ssm_a", w->ssm_a, DT_RANK);
    dump("final_norm", mdl.norm_weight, D_MODEL);
    dump("output_w", mdl.output_weight, D_MODEL * 248320);
    
    wubu_model_free(&mdl);
    return 0;
}
