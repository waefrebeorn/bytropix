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
    
    int D = D_MODEL;
    int vs = mdl.vocab_size;
    
    // Get BOS embedding
    float *bos_emb = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        const char *emb_path = "data/qwen36_embeddings_c.bin.raw";
        FILE *emb_f = fopen(emb_path, "rb");
        if (!emb_f) { printf("ERROR: cannot open embedding file\n"); return 1; }
        fseek(emb_f, 248044LL * D * sizeof(float), SEEK_SET);
        size_t n = fread(bos_emb, sizeof(float), D, emb_f);
        printf("Read BOS emb from file: %zu floats\n", n);
        fclose(emb_f);
    } else {
        memcpy(bos_emb, mdl.token_embd + 248044LL * D, D * sizeof(float));
        printf("Read BOS emb from memory\n");
    }
    
    // Embedding stats
    float emb_mean = 0, emb_std = 0;
    for (int i = 0; i < D; i++) emb_mean += bos_emb[i];
    emb_mean /= D;
    for (int i = 0; i < D; i++) emb_std += (bos_emb[i]-emb_mean)*(bos_emb[i]-emb_mean);
    emb_std = sqrtf(emb_std/D);
    printf("BOS emb: mean=%.4f std=%.4f first[0..4]=%.4f %.4f %.4f %.4f %.4f\n",
           emb_mean, emb_std, bos_emb[0], bos_emb[1], bos_emb[2], bos_emb[3], bos_emb[4]);
    
    // Save embedding for ref comparison (dump_ref_logits doesn't give us emb, but let's save anyway)
    FILE *f = fopen("/tmp/our_bos_emb.bin", "wb");
    fwrite(bos_emb, sizeof(float), D, f);
    fclose(f);
    printf("Saved BOS emb to /tmp/our_bos_emb.bin\n");
    
    // Now test just layer 0 (SSM) output
    float *normed = (float *)malloc(D * sizeof(float));
    wubu_rms_norm(1, 1, D, bos_emb, mdl.layers[0].attn_norm_weight, 1e-6f, normed);
    
    float *ssm_out = (float *)malloc(D * sizeof(float));
    memset(ssm_out, 0, D * sizeof(float));
    float *ssm_state = mdl.ssm_states + 0 * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    float *conv_state = mdl.conv_states + 0 * (CONV_KERNEL - 1) * CONV_DIM;
    wubu_ssm_forward(normed, 1, 1, &mdl.layers[0].ssm, ssm_state, conv_state, ssm_out, NULL, NULL);
    
    float ssm_mean = 0, ssm_std = 0;
    for (int i = 0; i < D; i++) ssm_mean += ssm_out[i];
    ssm_mean /= D;
    for (int i = 0; i < D; i++) ssm_std += (ssm_out[i]-ssm_mean)*(ssm_out[i]-ssm_mean);
    ssm_std = sqrtf(ssm_std/D);
    printf("L0 SSM output: mean=%.4f std=%.4f first[0..4]=%.4f %.4f %.4f %.4f %.4f\n",
           ssm_mean, ssm_std, ssm_out[0], ssm_out[1], ssm_out[2], ssm_out[3], ssm_out[4]);
    
    f = fopen("/tmp/our_l0_ssm.bin", "wb");
    fwrite(ssm_out, sizeof(float), D, f);
    fclose(f);
    printf("Saved L0 SSM output to /tmp/our_l0_ssm.bin\n");
    
    // Test output projection directly: take first token embedding, apply output weight
    float *test_logits = (float *)malloc(10 * sizeof(float));
    for (int j = 0; j < 10; j++) {
        double sum = 0.0;
        for (int k = 0; k < D; k++)
            sum += (double)normed[k] * (double)mdl.output_weight[j * D + k];
        test_logits[j] = (float)sum;
    }
    printf("Output projection test (first 10, using L0 normed):\n");
    for (int j = 0; j < 10; j++)
        printf("  [%d] = %.4f\n", j, test_logits[j]);
    
    free(bos_emb); free(normed); free(ssm_out); free(test_logits);
    wubu_model_free(&mdl);
    return 0;
}
