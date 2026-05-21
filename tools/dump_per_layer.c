/**
 * dump_per_layer.c — Dump CPU and GPU per-layer hidden states via DUMP_LAYER_DIR env.
 * Compare cos-sim at each layer to find error accumulation trajectory.
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>

int main(void) {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    mdl.enable_moe = true;
    
    int gpu_ok = 0;
    if (getenv("GPU")) {
        gpu_ok = wubu_model_gpu_init(&mdl, 4096, 256);
        fprintf(stderr, "GPU init: %s\n", gpu_ok ? "OK" : "FAILED");
    }
    if (!gpu_ok) { wubu_model_free(&mdl); return 1; }
    
    float embd[D_MODEL];
    if (mdl.token_embd)
        memcpy(embd, mdl.token_embd + 279LL * D_MODEL, D_MODEL * sizeof(float));
    else
        memset(embd, 0, D_MODEL * sizeof(float));
    
    // Save initial state
    size_t ssm_sz = (size_t)mdl.n_layers * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float);
    size_t conv_sz = (size_t)mdl.n_layers * (CONV_KERNEL - 1) * CONV_DIM * sizeof(float);
    float *saved_ssm = (float *)malloc(ssm_sz);
    float *saved_conv = (float *)malloc(conv_sz);
    memcpy(saved_ssm, mdl.ssm_states, ssm_sz);
    memcpy(saved_conv, mdl.conv_states, conv_sz);
    int saved_cache = mdl.gqa_cache_len;

    // Create dump dirs
    mkdir("/tmp/cpu_layers", 0755);
    mkdir("/tmp/gpu_layers", 0755);
    
    // === CPU forward (FORCE_CPU_MOE=1) ===
    setenv("FORCE_CPU_MOE", "1", 1);
    setenv("DUMP_LAYER_DIR", "/tmp/cpu_layers", 1);
    mdl.skip_output_proj = true;
    float hidden[D_MODEL];
    wubu_model_forward_from_embd(&mdl, embd, 1, 1, hidden);
    fprintf(stderr, "CPU done.\n");
    
    // Restore state
    memcpy(mdl.ssm_states, saved_ssm, ssm_sz);
    memcpy(mdl.conv_states, saved_conv, conv_sz);
    mdl.gqa_cache_len = saved_cache;
    for (int l = 0; l < mdl.n_layers; l++)
        if (mdl.layers[l].is_ssm)
            wubu_gpu_sync_ssm_state_to_gpu(mdl.gpu_ctx, l,
                mdl.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE,
                mdl.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM);
    
    // === GPU forward (no FORCE_CPU_MOE) ===
    unsetenv("FORCE_CPU_MOE");
    setenv("DUMP_LAYER_DIR", "/tmp/gpu_layers", 1);
    wubu_model_forward_from_embd(&mdl, embd, 1, 1, hidden);
    fprintf(stderr, "GPU done.\n");
    
    // === Compare per-layer ===
    printf("Layer | Cos-sim\n");
    printf("------|--------\n");
    float cpu_hidden[D_MODEL], gpu_hidden[D_MODEL];
    float *prev_cpu = (float *)calloc(D_MODEL, sizeof(float));
    float *prev_gpu = (float *)calloc(D_MODEL, sizeof(float));
    double running_dot = 0, running_n1 = 0, running_n2 = 0;
    
    for (int l = 0; l < mdl.n_layers; l++) {
        char fname[512];
        
        // Read CPU layer
        snprintf(fname, sizeof(fname), "/tmp/cpu_layers/our_layer_%d.bin", l);
        FILE *f = fopen(fname, "rb");
        if (!f) { printf("%5d | NO CPU DUMP\\n", l); continue; }
        fread(cpu_hidden, sizeof(float), D_MODEL, f);
        fclose(f);
        
        // Read GPU layer
        snprintf(fname, sizeof(fname), "/tmp/gpu_layers/our_layer_%d.bin", l);
        f = fopen(fname, "rb");
        if (!f) { printf("%5d | NO GPU DUMP\\n", l); continue; }
        fread(gpu_hidden, sizeof(float), D_MODEL, f);
        fclose(f);
        
        // Cos-sim for this layer's output
        double dot = 0, n1 = 0, n2 = 0;
        double layer_delta = 0, accum_delta_cpu = 0, accum_delta_gpu = 0;
        for (int i = 0; i < D_MODEL; i++) {
            dot += (double)cpu_hidden[i] * (double)gpu_hidden[i];
            n1  += (double)cpu_hidden[i] * (double)cpu_hidden[i];
            n2  += (double)gpu_hidden[i] * (double)gpu_hidden[i];
            layer_delta += (cpu_hidden[i] - gpu_hidden[i]) * (cpu_hidden[i] - gpu_hidden[i]);
            accum_delta_cpu += (cpu_hidden[i] - prev_cpu[i]) * (cpu_hidden[i] - prev_cpu[i]);
            accum_delta_gpu += (gpu_hidden[i] - prev_gpu[i]) * (gpu_hidden[i] - prev_gpu[i]);
        }
        
        // Running cumulative cos-sim
        for (int i = 0; i < D_MODEL; i++) {
            running_dot += (double)cpu_hidden[i] * (double)gpu_hidden[i];
            running_n1  += (double)cpu_hidden[i] * (double)cpu_hidden[i];
            running_n2  += (double)gpu_hidden[i] * (double)gpu_hidden[i];
        }
        
        double cos = dot / (sqrt(n1) * sqrt(n2));
        double run_cos = running_dot / (sqrt(running_n1) * sqrt(running_n2));
        double rms_err = sqrt(layer_delta / D_MODEL);
        double rms_cpu = sqrt(n1 / D_MODEL);
        double rel_err = rms_err / (rms_cpu + 1e-30f);
        
        printf("%5d (%s) | cos=%.6f run=%.6f | RMS_err=%.2e rel=%.4f\n",
               l, mdl.layers[l].is_ssm ? "SSM" : "GQA", cos, run_cos,
               rms_err, rel_err);
        
        memcpy(prev_cpu, cpu_hidden, D_MODEL * sizeof(float));
        memcpy(prev_gpu, gpu_hidden, D_MODEL * sizeof(float));
    }
    
    free(saved_ssm); free(saved_conv);
    free(prev_cpu); free(prev_gpu);
    wubu_model_free(&mdl);
    return 0;
}
