/* Save first 2048*16 floats of attn_qkv for layer 0 for comparison */
#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>

int main() {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    
    // Save first few values from attn_qkv for layer 0
    float *w = mdl.layers[0].ssm.attn_qkv_weight;
    
    // Print first 20
    printf("attn_qkv[0..19]: ");
    for (int i = 0; i < 20; i++) printf("%.10f ", w[i]);
    printf("\n");
    
    double m=0,s_v=0;
    for (int i = 0; i < 2048*16; i++) { m += w[i]; s_v += w[i]*w[i]; }
    int n = 2048*16;
    printf("Mean=%.10f Std=%.10f\n", m/n, sqrt(s_v/n - (m/n)*(m/n)));
    
    // Save to file for Python comparison
    FILE *f = fopen("/tmp/c_attn_qkv_part.bin", "wb");
    fwrite(w, sizeof(float), 2048*16, f);
    fclose(f);
    printf("Saved first 2048*16 floats to /tmp/c_attn_qkv_part.bin\n");
    
    // Also save attn_gate first 1000 values
    f = fopen("/tmp/c_attn_gate_part.bin", "wb");
    fwrite(mdl.layers[0].ssm.attn_gate_weight, sizeof(float), 1000, f);
    fclose(f);
    
    // Save conv1d weights (F32)
    f = fopen("/tmp/c_conv1d.bin", "wb");
    fwrite(mdl.layers[0].ssm.ssm_conv1d_weight, sizeof(float), 4*8192, f);
    fclose(f);
    
    // Save ssm_out first 1000 values
    f = fopen("/tmp/c_ssm_out_w.bin", "wb");
    fwrite(mdl.layers[0].ssm.ssm_out_weight, sizeof(float), 1000, f);
    fclose(f);
    
    printf("conv1d[0..9]: ");
    for (int i = 0; i < 10; i++) printf("%.10f ", mdl.layers[0].ssm.ssm_conv1d_weight[i]);
    printf("\n");
    
    wubu_model_free(&mdl);
    return 0;
}
