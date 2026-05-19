/* Dump ALL intermediates for SSM layer 0 for verification */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, path)) return 1;
    
    int D = D_MODEL;
    
    // Get BOS embedding
    float *bos_emb = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        const char *emb_path = "data/qwen36_embeddings_c.bin.raw";
        FILE *f = fopen(emb_path, "rb");
        fseek(f, 248044LL * D * sizeof(float), SEEK_SET);
        fread(bos_emb, sizeof(float), D, f);
        fclose(f);
    } else {
        memcpy(bos_emb, mdl.token_embd + 248044LL * D, D * sizeof(float));
    }
    
    // Layer 0 forward
    // Dump ALL weights of layer 0 SSM
    ssm_layer_weights *w = &mdl.layers[0].ssm;
    
    // attn_qkv.weight [D_MODEL, KEY_DIM*2+VALUE_DIM] = [2048, 8192]
    FILE *f = fopen("/tmp/our_l0_qkv.bin", "wb");
    fwrite(w->attn_qkv_weight, sizeof(float), D * (KEY_DIM*2+VALUE_DIM), f);
    fclose(f);
    
    // attn_gate.weight [D_MODEL, VALUE_DIM] = [2048, 4096]
    f = fopen("/tmp/our_l0_gate.bin", "wb");
    fwrite(w->attn_gate_weight, sizeof(float), D * VALUE_DIM, f);
    fclose(f);
    
    // ssm_beta.weight [D_MODEL, DT_RANK] = [2048, 32]
    f = fopen("/tmp/our_l0_beta.bin", "wb");
    fwrite(w->ssm_beta_weight, sizeof(float), D * DT_RANK, f);
    fclose(f);
    
    // ssm_alpha.weight [D_MODEL, DT_RANK] = [2048, 32]  
    f = fopen("/tmp/our_l0_alpha.bin", "wb");
    fwrite(w->ssm_alpha_weight, sizeof(float), D * DT_RANK, f);
    fclose(f);
    
    // ssm_dt.bias [DT_RANK] = [32]
    f = fopen("/tmp/our_l0_dt_bias.bin", "wb");
    fwrite(w->ssm_dt_bias, sizeof(float), DT_RANK, f);
    fclose(f);
    
    // ssm_a [DT_RANK] = [32] (-A_log)
    f = fopen("/tmp/our_l0_ssm_a.bin", "wb");
    fwrite(w->ssm_a, sizeof(float), DT_RANK, f);
    fclose(f);
    
    // ssm_conv1d.weight [CONV_KERNEL, CONV_DIM] = [4, 8192]
    f = fopen("/tmp/our_l0_conv.bin", "wb");
    fwrite(w->ssm_conv1d_weight, sizeof(float), CONV_KERNEL * CONV_DIM, f);
    fclose(f);
    
    // ssm_norm.weight [SSM_D_STATE] = [128]
    f = fopen("/tmp/our_l0_ssm_norm.bin", "wb");
    fwrite(w->ssm_norm_weight, sizeof(float), SSM_D_STATE, f);
    fclose(f);
    
    // ssm_out.weight [VALUE_DIM, D_MODEL] = [4096, 2048]
    f = fopen("/tmp/our_l0_ssm_out.bin", "wb");
    fwrite(w->ssm_out_weight, sizeof(float), VALUE_DIM * D_MODEL, f);
    fclose(f);
    
    // Also dump pre-attention RMSNorm weights
    f = fopen("/tmp/our_l0_attn_norm.bin", "wb");
    fwrite(mdl.layers[0].attn_norm_weight, sizeof(float), D, f);
    fclose(f);
    
    // BOS embedding
    f = fopen("/tmp/our_bos_emb.bin", "wb");
    fwrite(bos_emb, sizeof(float), D, f);
    fclose(f);
    
    printf("All weights dumped to /tmp/our_l0_*.bin\n");
    
    // Now run the layer and dump output
    float *normed = (float *)malloc(D * sizeof(float));
    wubu_rms_norm(1, 1, D, bos_emb, mdl.layers[0].attn_norm_weight, 1e-6f, normed);
    
    f = fopen("/tmp/our_l0_normed.bin", "wb");
    fwrite(normed, sizeof(float), D, f);
    fclose(f);
    
    float *ssm_out = (float *)malloc(D * sizeof(float));
    float *ssm_state = mdl.ssm_states;
    float *conv_state = mdl.conv_states;
    wubu_ssm_forward(normed, 1, 1, w, ssm_state, conv_state, ssm_out, NULL, NULL);
    
    f = fopen("/tmp/our_l0_ssm_out.bin", "wb");
    fwrite(ssm_out, sizeof(float), D, f);
    fclose(f);
    
    printf("L0 outputs saved\n");
    
    // Dump ALL tensor info from GGUF for this layer
    printf("\nGGUF tensor info for layer 0:\n");
    gguf_ctx *ctx = mdl.gguf_ctx;
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (strncmp(ctx->tensors[i].name, "blk.0.", 6) == 0) {
            gguf_tensor_info *ti = &ctx->tensors[i];
            printf("  %s: dims=[%ld,%ld,%ld,%ld] type=%d n_dims=%d\n",
                   ti->name, (long)ti->dims[0], (long)ti->dims[1], (long)ti->dims[2], (long)ti->dims[3],
                   ti->ggml_type, ti->n_dims);
        }
    }
    
    free(bos_emb); free(normed); free(ssm_out);
    wubu_model_free(&mdl);
    return 0;
}
