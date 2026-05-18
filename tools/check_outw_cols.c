/* Verify specific columns of output weight between our dequant and Python */
#include "wubu_model.h"
#include <stdio.h>
#include <math.h>

int main() {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    
    int D = D_MODEL;
    float *w = mdl.output_weight;
    
    // Check columns for specific token IDs
    int tokens[] = {0, 220, 264, 84944, 55073, 248044, 248319};
    for (int ti = 0; ti < 6; ti++) {
        int tok = tokens[ti];
        double m=0,s=0;
        for (int k = 0; k < D; k++) {
            float v = w[tok * D + k];
            m += v; s += v*v;
        }
        m /= D; s = sqrt(s/D - m*m);
        printf("Token %d col: mean=%.8f std=%.8f first5=", tok, m, s);
        for (int k = 0; k < 5; k++) printf("%.8f ", w[tok * D + k]);
        printf("\n");
    }
    
    // Save column 0 for Python comparison
    FILE *f = fopen("/tmp/c_outw_col0.bin", "wb");
    fwrite(w, sizeof(float), D, f);
    fclose(f);
    
    // Save column 220 for Python comparison
    f = fopen("/tmp/c_outw_col220.bin", "wb");
    fwrite(w + 220LL * D, sizeof(float), D, f);
    fclose(f);
    
    // Save column 84944 for Python comparison
    f = fopen("/tmp/c_outw_col84944.bin", "wb");
    fwrite(w + 84944LL * D, sizeof(float), D, f);
    fclose(f);
    
    printf("\nSaved columns for Python verification\n");
    
    wubu_model_free(&mdl);
    return 0;
}
