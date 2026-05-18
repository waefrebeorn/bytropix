/* Compare final hidden states: ours vs reference, before output projection */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include "llama.h"

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    // ---- Our model forward, save hidden state after final norm ----
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, path)) return 1;
    mdl.enable_moe = true;
    
    int D = D_MODEL;
    float *x = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        const char *emb_path = "data/qwen36_embeddings_c.bin.raw";
        FILE *emb_f = fopen(emb_path, "rb");
        if (!emb_f) return 1;
        fseek(emb_f, 248044LL * D * sizeof(float), SEEK_SET);
        fread(x, sizeof(float), D, emb_f);
        fclose(emb_f);
    } else {
        memcpy(x, mdl.token_embd + 248044LL * D, D * sizeof(float));
    }
    
    // Forward pass - run inside wubu_model but we need to capture hidden state...
    // Easiest: modify the forward to also save hidden state
    // For now, let's use the existing forward and instead:
    // 1. Run wubu_model_forward_from_embd to get logits
    // 2. Compare logits vs reference
    
    int vs = mdl.vocab_size;
    float *logits = (float *)malloc(vs * sizeof(float));
    wubu_model_forward_from_embd(&mdl, x, 1, 1, logits);
    
    // Compare vs reference logits
    float *ref = (float *)malloc(vs * sizeof(float));
    FILE *f = fopen("/tmp/ref_logits_fresh.bin", "rb");
    if (!f) return 1;
    fread(ref, sizeof(float), vs, f);
    fclose(f);
    
    double dot = 0, n_our = 0, n_ref = 0;
    for (int i = 0; i < vs; i++) {
        dot += (double)logits[i] * (double)ref[i];
        n_our += (double)logits[i] * (double)logits[i];
        n_ref += (double)ref[i] * (double)ref[i];
    }
    double cos = dot / (sqrt(n_our) * sqrt(n_ref));
    printf("Full logits cos-sim: %.6f (n_our=%.2f n_ref=%.2f)\n", cos, sqrt(n_our), sqrt(n_ref));
    
    // Now get OUR hidden state by doing ONE MORE FORWARD but intercepting
    // Actually, since we can't intercept hidden state, let's compare the output
    // projection weight values directly.
    
    // Dump first 5 vs reference logit values
    printf("Our logits first 10: ");
    for (int i = 0; i < 10; i++) printf("%.4f ", logits[i]);
    printf("\n");
    printf("Ref logits first 10: ");
    for (int i = 0; i < 10; i++) printf("%.4f ", ref[i]);
    printf("\n");
    
    // Compare output weight mean magnitude
    double wm_our = 0;
    for (int i = 0; i < D * 10; i++) wm_our += fabs(mdl.output_weight[i]);
    printf("Output weight first 10*2048 mean abs: %.6f\n", wm_our / (D * 10));
    
    free(x); free(logits); free(ref);
    wubu_model_free(&mdl);
    return 0;
}
