/**
 * Dump shared expert before and after gating.
 */
#include "wubu_model.h"
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void dump(const char *path, const float *data, int n) {
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(data, sizeof(float), n, f); fclose(f); }
}

int main(void) {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    
    float D = D_MODEL;
    float *x = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        fseek(f, 248044LL * (int64_t)D * sizeof(float), SEEK_SET);
        fread(x, sizeof(float), D, f); fclose(f);
    }
    
    wubu_layer_t *layer = &mdl.layers[0];
    float *normed = (float *)malloc(D * sizeof(float));
    wubu_rms_norm(1, 1, D, x, layer->attn_norm_weight, 1e-6f, normed);
    
    float *ssm_state = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
    float *conv_state = (float *)calloc((CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
    float *attn_out = (float *)malloc(D * sizeof(float));
    wubu_ssm_forward(normed, 1, 1, &layer->ssm, ssm_state, conv_state, attn_out, NULL, NULL);
    for (int i = 0; i < D; i++) x[i] += attn_out[i];
    
    float *normed2 = (float *)malloc(D * sizeof(float));
    wubu_rms_norm(1, 1, D, x, layer->post_attn_norm_weight, 1e-6f, normed2);
    
    moe_weights_t moe;
    if (!wubu_moe_load_layer(mdl.gguf_ctx, 0, &moe)) return 1;
    
    // === ROUTED EXPERTS ===
    float scores[256];
    wubu_moe_router(normed2, 1, 1, moe.ffn_gate_inp, scores);
    
    float max_s = scores[0];
    for (int e = 1; e < 256; e++) if (scores[e] > max_s) max_s = scores[e];
    float sum_exp = 0;
    for (int e = 0; e < 256; e++) sum_exp += expf(scores[e] - max_s);
    float inv_sum = 1.0f / sum_exp;
    float probs[256];
    for (int e = 0; e < 256; e++) probs[e] = expf(scores[e] - max_s) * inv_sum;
    
    int indices[8]; float weights[8];
    for (int k = 0; k < 8; k++) { indices[k]=k; weights[k]=probs[k]; }
    for (int i = 0; i < 7; i++) for (int j = i+1; j < 8; j++)
        if (weights[i] > weights[j]) {
            float tw=weights[i]; weights[i]=weights[j]; weights[j]=tw;
            int ti=indices[i]; indices[i]=indices[j]; indices[j]=ti;
        }
    for (int e = 8; e < 256; e++) {
        if (probs[e] > weights[0]) {
            weights[0] = probs[e]; indices[0] = e;
            int pos = 0;
            while (pos+1 < 8 && weights[pos] > weights[pos+1]) {
                float tw=weights[pos]; weights[pos]=weights[pos+1]; weights[pos+1]=tw;
                int ti=indices[pos]; indices[pos]=indices[pos+1]; indices[pos+1]=ti;
                pos++;
            }
        }
    }
    float sum_w = 0;
    for (int k = 0; k < 8; k++) sum_w += weights[k];
    float is = 1.0f/sum_w;
    for (int k = 0; k < 8; k++) weights[k] *= is;
    
    printf("Normalized weights: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
           weights[0], weights[1], weights[2], weights[3],
           weights[4], weights[5], weights[6], weights[7]);
    
    // === ROUTED ONLY ===
    float routed_out[D_MODEL]; memset(routed_out, 0, sizeof(routed_out));
    for (int k = 0; k < 8; k++) {
        int e = indices[k]; float wgt = weights[k];
        const float *gw = moe.ffn_gate_exps + (int64_t)e * D_MODEL * D_FF;
        const float *uw = moe.ffn_up_exps   + (int64_t)e * D_MODEL * D_FF;
        const float *dw = moe.ffn_down_exps  + (int64_t)e * D_FF * D_MODEL;
        
        float expert_out[D_MODEL]; memset(expert_out, 0, sizeof(expert_out));
        float gate_out[D_FF], up_out[D_FF], act[D_FF];
        for (int j = 0; j < D_FF; j++) {
            double s = 0;
            for (int kk = 0; kk < D_MODEL; kk++) s += normed2[kk] * gw[kk + j * D_MODEL];
            gate_out[j] = s;
        }
        for (int j = 0; j < D_FF; j++) {
            double s = 0;
            for (int kk = 0; kk < D_MODEL; kk++) s += normed2[kk] * uw[kk + j * D_MODEL];
            up_out[j] = s;
        }
        for (int j = 0; j < D_FF; j++) {
            float g = gate_out[j]; float silu = (g < -80) ? 0 : g / (1+expf(-g));
            act[j] = silu * up_out[j];
        }
        for (int j = 0; j < D_MODEL; j++) {
            double s = 0;
            for (int kk = 0; kk < D_FF; kk++) s += act[kk] * dw[kk + j * D_FF];
            expert_out[j] = s;
        }
        for (int j = 0; j < D_MODEL; j++) routed_out[j] += wgt * expert_out[j];
    }
    
    // === SHARED EXPERT: Before gating ===
    float shared_ungated[D_MODEL]; memset(shared_ungated, 0, sizeof(shared_ungated));
    int SFF = SHARED_D_FF;
    for (int j = 0; j < SFF; j++) {
        double s = 0;
        for (int kk = 0; kk < D_MODEL; kk++) s += normed2[kk] * moe.ffn_gate_shexp[kk + j * D_MODEL];
        float g = s;
        float silu = (g < -80) ? 0 : g / (1+expf(-g));
        double us = 0;
        for (int kk = 0; kk < D_MODEL; kk++) us += normed2[kk] * moe.ffn_up_shexp[kk + j * D_MODEL];
        float act = silu * us;
        for (int kk = 0; kk < D_MODEL; kk++)
            shared_ungated[kk] += act * moe.ffn_down_shexp[j + kk * SFF];
    }
    
    // === SHARED EXPERT gate ===
    double gate_val = 0;
    for (int kk = 0; kk < D_MODEL; kk++)
        gate_val += normed2[kk] * moe.ffn_gate_inp_shexp[kk];
    float gate_sig = 1.0f / (1.0f + expf(-(float)gate_val));
    
    // Shared output after gating
    float shared_gated[D_MODEL];
    for (int j = 0; j < D_MODEL; j++) shared_gated[j] = shared_ungated[j] * gate_sig;
    
    // Total MoE output
    float total[D_MODEL];
    for (int j = 0; j < D_MODEL; j++) total[j] = routed_out[j] + shared_gated[j];
    
    printf("\nShared gate: sigmoid = %.6f\n", gate_sig);
    
    float rm=0, rs=0, sum=0, ss=0, sum2=0, ss2=0, tm=0, ts=0;
    for (int j = 0; j < D_MODEL; j++) {
        rm += routed_out[j]; rs += routed_out[j]*routed_out[j];
        sum += shared_ungated[j]; ss += shared_ungated[j]*shared_ungated[j];
        sum2 += shared_gated[j]; ss2 += shared_gated[j]*shared_gated[j];
        tm += total[j]; ts += total[j]*total[j];
    }
    printf("Routed:       mean=%+.6f std=%.6f  norm=%.4f\n", rm/D_MODEL, sqrtf(rs/D_MODEL-(rm/D_MODEL)*(rm/D_MODEL)), sqrtf(rs));
    printf("Shared (ung): mean=%+.6f std=%.6f  norm=%.4f\n", sum/D_MODEL, sqrtf(ss/D_MODEL-(sum/D_MODEL)*(sum/D_MODEL)), sqrtf(ss));
    printf("Shared (gat): mean=%+.6f std=%.6f  norm=%.4f\n", sum2/D_MODEL, sqrtf(ss2/D_MODEL-(sum2/D_MODEL)*(sum2/D_MODEL)), sqrtf(ss2));
    printf("Total:        mean=%+.6f std=%.6f  norm=%.4f\n", tm/D_MODEL, sqrtf(ts/D_MODEL-(tm/D_MODEL)*(tm/D_MODEL)), sqrtf(ts));
    
    // Check: what's the L2 norm of the MoE input?
    float im=0, isq=0;
    for (int j = 0; j < D_MODEL; j++) { im += normed2[j]; isq += normed2[j]*normed2[j]; }
    float input_norm = sqrtf(isq);
    printf("MoE input:    norm=%.4f\n", input_norm);
    
    // Ratio of total output norm to input norm
    printf("Output/Input norm ratio: %.4f\n", sqrtf(ts) / input_norm);
    
    wubu_moe_free_layer(&moe);
    wubu_model_free(&mdl);
    free(x); free(normed); free(ssm_state); free(conv_state);
    free(attn_out); free(normed2);
    return 0;
}
