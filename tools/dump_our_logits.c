/* Dump our model's logits + last hidden state for a single BOS token */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, path)) return 1;
    mdl.enable_moe = true; // Enable MoE for correct output
    
    int vs = mdl.vocab_size;
    int D = D_MODEL;
    
    // Get embedding for BOS token (248044)
    float *x = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        const char *emb_path = "data/qwen36_embeddings_c.bin.raw";
        FILE *emb_f = fopen(emb_path, "rb");
        float *all_emb = (float *)malloc((int64_t)vs * D * sizeof(float));
        fread(all_emb, sizeof(float), (int64_t)vs * D, emb_f);
        fclose(emb_f);
        memcpy(x, all_emb + 248044LL * D, D * sizeof(float));
        free(all_emb);
    } else {
        memcpy(x, mdl.token_embd + 248044LL * D, D * sizeof(float));
    }
    
    float *logits = (float *)malloc(vs * sizeof(float));
    wubu_model_forward_from_embd(&mdl, x, 1, 1, logits);
    
    // Dump first 100 logits
    FILE *f = fopen("/tmp/our_logits.bin", "wb");
    fwrite(logits, sizeof(float), vs, f);
    fclose(f);
    
    // Find top-5
    int top[5] = {0}; float tv[5] = {-1e30,-1e30,-1e30,-1e30,-1e30};
    for (int i = 0; i < vs; i++) {
        if (logits[i] > tv[4]) {
            tv[4] = logits[i]; top[4] = i;
            for (int k = 3; k >= 0; k--) {
                if (tv[k] < tv[k+1]) {
                    float tmp = tv[k]; tv[k] = tv[k+1]; tv[k+1] = tmp;
                    int ti = top[k]; top[k] = top[k+1]; top[k+1] = ti;
                }
            }
        }
    }
    printf("Our top-5 tokens:\n");
    for (int k = 0; k < 5; k++)
        printf("  [%d] val=%.4f\n", top[k], tv[k]);
    printf("Logits stats: mean=%.4f std=%.4f max=%.4f min=%.4f\n",
           logits[0], logits[1], tv[0], tv[4]);
    
    free(x); free(logits);
    wubu_model_free(&mdl);
    printf("Dumped /tmp/our_logits.bin\n");
    return 0;
}
