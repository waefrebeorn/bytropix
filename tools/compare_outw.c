/* Save first token of output.weight from our dequant for comparison */
#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, path)) return 1;
    
    int D = D_MODEL;
    float *buf = (float *)malloc(D * sizeof(float));
    for (int k = 0; k < D; k++)
        buf[k] = mdl.output_weight[k];  // first token of output weight
    FILE *f = fopen("/tmp/our_outw_token0.bin", "wb");
    fwrite(buf, sizeof(float), D, f);
    fclose(f);
    printf("First 10: ");
    for (int i = 0; i < 10; i++) printf("%.8f ", buf[i]);
    printf("\n");
    double m=0,s=0;
    for(int i=0;i<D;i++){m+=buf[i];s+=buf[i]*buf[i];}
    printf("Mean=%.8f Std=%.8f\n", m/D, sqrt(s/D-(m/D)*(m/D)));
    
    free(buf);
    wubu_model_free(&mdl);
    return 0;
}
