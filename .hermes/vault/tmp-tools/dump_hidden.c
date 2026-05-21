/**
 * Simple embedding check after GPU init.
 */
#include "wubu_model.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    
    // Check embedding BEFORE GPU init
    float e0[D_MODEL];
    memcpy(e0, mdl.token_embd + 279LL * D_MODEL, D_MODEL * sizeof(float));
    printf("Before GPU init: embd[0]=%.6f embd[1]=%.6f embd[100]=%.6f\n", e0[0], e0[1], e0[100]);
    
    // GPU init
    int ok = wubu_model_gpu_init(&mdl, 4096, 256);
    printf("GPU init: %s\n", ok ? "OK" : "FAILED");
    
    // Check embedding AFTER GPU init
    float e1[D_MODEL];
    memcpy(e1, mdl.token_embd + 279LL * D_MODEL, D_MODEL * sizeof(float));
    printf("After GPU init: embd[0]=%.6f embd[1]=%.6f embd[100]=%.6f\n", e1[0], e1[1], e1[100]);
    
    // Compare
    int same = memcmp(e0, e1, D_MODEL * sizeof(float)) == 0;
    printf("Embeddings identical: %s\n", same ? "YES" : "NO");
    if (!same) {
        int diff_count = 0;
        for (int i = 0; i < D_MODEL; i++) {
            if (fabsf(e0[i] - e1[i]) > 1e-10f) {
                if (diff_count < 5) printf("  diff[%d]: %.10f vs %.10f\n", i, e0[i], e1[i]);
                diff_count++;
            }
        }
        printf("Total differences: %d/%d\n", diff_count, D_MODEL);
    }
    
    wubu_model_free(&mdl);
    return 0;
}
