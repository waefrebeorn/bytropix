#!/usr/bin/env python3
"""
Compute shared expert output for layer 0 using our MoE input.
Use Python's gguf library to get the weights (via raw file extraction + our C dequant)
or use the already-loaded model.

Goal: verify that the shared expert computation is correct.
If shared expert is wrong = fix there.
If shared expert is right = bug is in routed experts.
"""
import struct
import numpy as np
import ctypes
import os

# Load C library for dequant
lib = ctypes.CDLL(os.path.expanduser("~/bytropix/src/gguf_reader.o"))

# Read MoE input
moe_input = np.frombuffer(open("/tmp/dbg_moe_input.bin","rb").read(), dtype=np.float32)

D_MODEL = 2048
SHARED_D_FF = 512

# ==========================================
# APPROACH: Write a C program that:
# 1. Loads shared expert weights from our model
# 2. Computes shared expert output
# 3. Dumps to file
# 
# Then we can compare with our wubu_moe_forward output
# ==========================================

# Actually, let me just use the dump_moe_layer0 tool we already have
# It already dumps /tmp/dbg_moe_output.bin (the full MoE output including shared)
# We need to also dump the ROUTED-ONLY and SHARED-ONLY outputs

print("We need to dump routed-only and shared-only MoE outputs separately.")
print("Modify the C tool to output these separately.")

# Let me modify the C tool to dump intermediate MoE components
code = '''
/**
 * Dump routed-only and shared-only MoE outputs for layer 0.
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
    
    float *x = (float *)malloc(D_MODEL * sizeof(float));
    if (mdl.use_embedding_file) {
        FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (!f) return 1;
        fseek(f, 248044LL * D_MODEL * sizeof(float), SEEK_SET);
        fread(x, sizeof(float), D_MODEL, f); fclose(f);
    }
    
    // Run SSM to get MoE input
    wubu_layer_t *layer = &mdl.layers[0];
    float *normed = (float *)malloc(D_MODEL * sizeof(float));
    wubu_rms_norm(1, 1, D_MODEL, x, layer->attn_norm_weight, 1e-6f, normed);
    
    float *ssm_state = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
    float *conv_state = (float *)calloc((CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
    float *attn_out = (float *)malloc(D_MODEL * sizeof(float));
    wubu_ssm_forward(normed, 1, 1, &layer->ssm, ssm_state, conv_state, attn_out);
    
    for (int i = 0; i < D_MODEL; i++) x[i] += attn_out[i];
    
    float *normed2 = (float *)malloc(D_MODEL * sizeof(float));
    wubu_rms_norm(1, 1, D_MODEL, x, layer->post_attn_norm_weight, 1e-6f, normed2);
    dump("/tmp/dbg_moe_input.bin", normed2, D_MODEL);
    
    // Load MoE weights
    moe_weights_t moe;
    if (!wubu_moe_load_layer(mdl.gguf_ctx, 0, &moe)) {
        printf("Failed to load MoE\\n"); return 1;
    }
    
    // ===== ROUTED EXPERTS ONLY =====
    // Router
    float scores[256];
    wubu_moe_router(normed2, 1, 1, moe.ffn_gate_inp, scores);
    
    // Softmax + top-8 + normalize
    float max_s = scores[0];
    for (int e = 1; e < 256; e++) if (scores[e] > max_s) max_s = scores[e];
    float sum_exp = 0;
    for (int e = 0; e < 256; e++) sum_exp += expf(scores[e] - max_s);
    float inv_sum = 1.0f / (sum_exp + 1e-30f);
    float probs[256];
    for (int e = 0; e < 256; e++) probs[e] = expf(scores[e] - max_s) * inv_sum;
    
    int indices[8];
    float weights[8];
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
    if (sum_w > 1e-30f) { float is = 1.0f/sum_w; for (int k = 0; k < 8; k++) weights[k] *= is; }
    
    // Compute routed experts only (accumulate into routed_out)
    float routed_out[D_MODEL];
    memset(routed_out, 0, sizeof(routed_out));
    float temp[D_FF * 3];
    
    for (int k = 0; k < 8; k++) {
        int e = indices[k];
        float wgt = weights[k];
        if (wgt < 1e-30f) continue;
        
        const float *gw = moe.ffn_gate_exps + (int64_t)e * D_MODEL * D_FF;
        const float *uw = moe.ffn_up_exps   + (int64_t)e * D_MODEL * D_FF;
        const float *dw = moe.ffn_down_exps  + (int64_t)e * D_FF * D_MODEL;
        
        float expert_out[D_MODEL];
        memset(expert_out, 0, sizeof(expert_out));
        
        // gate = x @ gate_w
        float gate_out[D_FF];
        for (int j = 0; j < D_FF; j++) {
            double s = 0;
            for (int kk = 0; kk < D_MODEL; kk++)
                s += normed2[kk] * gw[kk + j * D_MODEL];
            gate_out[j] = s;
        }
        // up = x @ up_w
        float up_out[D_FF];
        for (int j = 0; j < D_FF; j++) {
            double s = 0;
            for (int kk = 0; kk < D_MODEL; kk++)
                s += normed2[kk] * uw[kk + j * D_MODEL];
            up_out[j] = s;
        }
        // act = SiLU(gate) * up
        float act[D_FF];
        for (int j = 0; j < D_FF; j++) {
            float g = gate_out[j];
            float silu = (g < -80.0f) ? 0.0f : g / (1.0f + expf(-g));
            act[j] = silu * up_out[j];
        }
        // output = act @ down_w
        for (int j = 0; j < D_MODEL; j++) {
            double s = 0;
            for (int kk = 0; kk < D_FF; kk++)
                s += act[kk] * dw[kk + j * D_FF];
            expert_out[j] = s;
        }
        
        // Weighted sum
        for (int j = 0; j < D_MODEL; j++)
            routed_out[j] += wgt * expert_out[j];
    }
    dump("/tmp/dbg_routed_only.bin", routed_out, D_MODEL);
    
    // ===== SHARED EXPERT ONLY =====
    float shared_out[D_MODEL];
    memset(shared_out, 0, sizeof(shared_out));
    
    // gate = x @ gate_shexp
    float shared_gate_out[SHARED_D_FF];
    for (int j = 0; j < SHARED_D_FF; j++) {
        double s = 0;
        for (int kk = 0; kk < D_MODEL; kk++)
            s += normed2[kk] * moe.ffn_gate_shexp[kk + j * D_MODEL];
        shared_gate_out[j] = s;
    }
    // up = x @ up_shexp
    float shared_up_out[SHARED_D_FF];
    for (int j = 0; j < SHARED_D_FF; j++) {
        double s = 0;
        for (int kk = 0; kk < D_MODEL; kk++)
            s += normed2[kk] * moe.ffn_up_shexp[kk + j * D_MODEL];
        shared_up_out[j] = s;
    }
    // act = SiLU(gate) * up
    float shared_act[SHARED_D_FF];
    for (int j = 0; j < SHARED_D_FF; j++) {
        float g = shared_gate_out[j];
        float silu = (g < -80.0f) ? 0.0f : g / (1.0f + expf(-g));
        shared_act[j] = silu * shared_up_out[j];
    }
    // output = act @ down_shexp
    for (int j = 0; j < D_MODEL; j++) {
        double s = 0;
        for (int kk = 0; kk < SHARED_D_FF; kk++)
            s += shared_act[kk] * moe.ffn_down_shexp[kk + j * SHARED_D_FF];
        shared_out[j] = s;
    }
    
    // Apply shared expert gate: sigmoid(x @ ffn_gate_inp_shexp)
    if (moe.ffn_gate_inp_shexp) {
        double gate_val = 0;
        for (int kk = 0; kk < D_MODEL; kk++)
            gate_val += normed2[kk] * moe.ffn_gate_inp_shexp[kk];
        float gate_sig = 1.0f / (1.0f + expf(-(float)gate_val));
        for (int j = 0; j < D_MODEL; j++)
            shared_out[j] *= gate_sig;
    }
    dump("/tmp/dbg_shared_only.bin", shared_out, D_MODEL);
    
    // Combined
    float combined_out[D_MODEL];
    for (int j = 0; j < D_MODEL; j++)
        combined_out[j] = routed_out[j] + shared_out[j];
    dump("/tmp/dbg_combined.bin", combined_out, D_MODEL);
    
    // Statistics
    {
        float rm=0, rs=0, sm=0, ss=0;
        for (int i=0;i<D_MODEL;i++){rm+=routed_out[i];rs+=routed_out[i]*routed_out[i];sm+=shared_out[i];ss+=shared_out[i]*shared_out[i];}
        printf("Routed:  mean=%.6f std=%.6f\\n", rm/D_MODEL, sqrtf(rs/D_MODEL - (rm/D_MODEL)*(rm/D_MODEL)));
        printf("Shared:  mean=%.6f std=%.6f\\n", sm/D_MODEL, sqrtf(ss/D_MODEL - (sm/D_MODEL)*(sm/D_MODEL)));
        
        // Compare with full MoE output
        float full[D_MODEL];
        wubu_moe_forward(normed2, 1, 1, &moe, full);
        float fm=0, fs=0;
        for (int i=0;i<D_MODEL;i++){fm+=full[i];fs+=full[i]*full[i];}
        printf("Full:    mean=%.6f std=%.6f\\n", fm/D_MODEL, sqrtf(fs/D_MODEL - (fm/D_MODEL)*(fm/D_MODEL)));
        
        // Check combined vs full
        double max_d=0;
        for (int i=0;i<D_MODEL;i++){double d=fabs(combined_out[i]-full[i]);if(d>max_d)max_d=d;}
        printf("Combined vs Full: max_diff=%.10f\\n", max_d);
    }
    
    wubu_moe_free_layer(&moe);
    free(x); free(normed); free(ssm_state); free(conv_state);
    free(attn_out); free(normed2);
    return 0;
}
'''

with open('/tmp/dump_routed_shared.c', 'w') as f:
    f.write(code)
print("C code written to /tmp/dump_routed_shared.c")

# Now compile and run
import subprocess
ret = subprocess.run(
    "cd /home/wubu/bytropix && gcc -O3 -march=native -ffast-math -fopenmp -I include "
    "-o /tmp/dump_routed_shared /tmp/dump_routed_shared.c "
    "src/wubu_model.o src/wubu_ssm.o src/wubu_ssm_chunked.o src/wubu_mobius.o "
    "src/wubu_nested_ssm.o src/wubu_nested_ssm_backward.o src/wubu_moe.o "
    "src/wubu_moe_backward.o src/wubu_moe_hyperbolic.o src/wubu_poincare_ssm_backward.o "
    "src/wubu_poincare_gqa.o src/wubu_poincare_gqa_backward.o src/wubu_mobius_linear.o "
    "src/wubu_hyperbolic_output_proj.o src/wubu_vision.o src/gguf_reader.o "
    "src/qlearner.o src/rsgd.o src/wubu_tst.o src/dequant_iq2_xxs.o "
    "-lm -fopenmp",
    shell=True, capture_output=True, text=True, timeout=30,
    cwd="/home/wubu/bytropix"
)
print(ret.stdout[-200:] if len(ret.stdout) > 200 else ret.stdout)
print(ret.stderr[-200:] if len(ret.stderr) > 200 else ret.stderr)

if ret.returncode == 0:
    ret2 = subprocess.run(
        "timeout 120 /tmp/dump_routed_shared",
        shell=True, capture_output=True, text=True, timeout=130,
        cwd="/home/wubu/bytropix"
    )
    print(ret2.stdout)
    if ret2.stderr:
        print("STDERR:", ret2.stderr[-500:])
