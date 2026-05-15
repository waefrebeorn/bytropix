// Run model forward and dump per-layer hidden state stats
#include "wubu_model.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    wubu_model_t model;
    if (!wubu_model_init(&model, path)) return 1;
    
    // Get embeddings (BOS token)
    gguf_ctx *ctx = model.gguf_ctx;
    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    int64_t ne = t->dims[0] * t->dims[1];
    int vs = (int)(ne / D_MODEL);
    float *embd = malloc(ne * sizeof(float));
    gguf_read_tensor_f32(ctx, t, embd, ne);
    
    // Use token 0 as input (just need something)
    float x[2048];
    memcpy(x, embd, 2048 * sizeof(float));
    
    // Run model forward, dump per-layer stats
    // We have to modify wubu_model_forward_from_embd to add dumps
    // OR we can manually run the layer loop like infer_text.c does
    
    // Actually, let's just manually run layer by layer
    float *h = (float *)malloc(1 * D_MODEL * sizeof(float));
    memcpy(h, x, D_MODEL * sizeof(float));
    
    for (int l = 0; l < model.n_layers; l++) {
        wubu_layer_t *layer = &model.layers[l];
        
        float *normed = (float *)malloc(D_MODEL * sizeof(float));
        wubu_rms_norm(1, 1, D_MODEL, h, layer->attn_norm_weight, 1e-6f, normed);
        
        float *attn_out = (float *)malloc(D_MODEL * sizeof(float));
        memset(attn_out, 0, D_MODEL * sizeof(float));
        
        if (layer->is_ssm) {
            float *ssm_state = model.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
            float *conv_state = model.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            wubu_ssm_forward(normed, 1, 1, &layer->ssm, ssm_state, conv_state, attn_out);
        } else {
            wubu_gqa_forward(normed, 1, 1, &layer->gqa, attn_out);
        }
        
        // Check for NaN/Inf
        int nan_count = 0, inf_count = 0;
        float maxv = -1e30, minv = 1e30, sum = 0;
        for (int i = 0; i < D_MODEL; i++) {
            if (isnan(attn_out[i])) nan_count++;
            if (isinf(attn_out[i])) inf_count++;
            if (attn_out[i] > maxv) maxv = attn_out[i];
            if (attn_out[i] < minv) minv = attn_out[i];
            sum += fabsf(attn_out[i]);
        }
        float mean_abs = sum / D_MODEL;
        
        printf("L%02d %s: attn_out mean_abs=%.4f range=[%.2f,%.2f] nan=%d inf=%d\n",
               l, layer->is_ssm ? "SSM" : "GQA",
               mean_abs, minv, maxv, nan_count, inf_count);
        
        // Residual
        for (int i = 0; i < D_MODEL; i++) h[i] += attn_out[i];
        
        // Post-attn norm + FFN pass-through
        wubu_rms_norm(1, 1, D_MODEL, h, layer->post_attn_norm_weight, 1e-6f, normed);
        // No MoE loaded, so ffn_out = normed (pass-through in wubu_moe_forward with !loaded)
        for (int i = 0; i < D_MODEL; i++) h[i] += normed[i];
        
        // Dump residual stats
        float r_max = -1e30, r_min = 1e30, r_sum = 0;
        for (int i = 0; i < D_MODEL; i++) {
            if (h[i] > r_max) r_max = h[i];
            if (h[i] < r_min) r_min = h[i];
            r_sum += fabsf(h[i]);
        }
        if (l < 5 || l >= model.n_layers - 2)
            printf("       residual: mean_abs=%.4f range=[%.2f,%.2f]\n",
                   r_sum/D_MODEL, r_min, r_max);
        
        free(normed);
        free(attn_out);
    }
    
    // Final norm
    float *final_normed = (float *)malloc(D_MODEL * sizeof(float));
    wubu_rms_norm(1, 1, D_MODEL, h, model.norm_weight, 1e-6f, final_normed);
    
    // Output projection
    float *logits = (float *)malloc(vs * sizeof(float));
    for (int j = 0; j < vs; j++) {
        double sum = 0.0;
        for (int k = 0; k < D_MODEL; k++)
            sum += (double)final_normed[k] * (double)model.output_weight[j * D_MODEL + k];
        logits[j] = (float)sum;
    }
    
    // Top-5
    int top[5] = {0}; float tv[5] = {-1e30,-1e30,-1e30,-1e30,-1e30};
    for (int j = 0; j < vs; j++) {
        if (logits[j] > tv[4]) {
            tv[4] = logits[j]; top[4] = j;
            for (int k = 3; k >= 0; k--) {
                if (tv[k] < tv[k+1]) {
                    float tmp = tv[k]; tv[k] = tv[k+1]; tv[k+1] = tmp;
                    int ti = top[k]; top[k] = top[k+1]; top[k+1] = ti;
                }
            }
        }
    }
    printf("\nTop-5:\n");
    // Decode tokens
    #include "wubu_tokenizer.h"
    wubu_tokenizer_t tok;
    wubu_tokenizer_init(&tok, path);
    char buf[256];
    for (int k = 0; k < 5; k++) {
        wubu_tokenizer_decode(&tok, top+k, 1, buf, 255);
        printf("  [%d]='%s'(%.2f)\n", top[k], buf, tv[k]);
    }
    
    free(logits); free(final_normed); free(h); free(embd);
    wubu_tokenizer_free(&tok);
    wubu_model_free(&model);
    printf("=== PASS ===\n");
    return 0;
}
