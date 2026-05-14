/**
 * train_gpu.c — Phase 3.5: GPU-Accelerated Training
 *
 * Hybrid: GPU for attention/SSM forward, CPU for norms + output projection.
 * Uses existing GPU kernels (wubu_cuda_rms_norm, gpu_ssm_forward, etc.)
 *
 * Usage: ./train_gpu [model.gguf] [corpus.bin] [steps]
 *   LR=0.001    learning rate
 *   B=1 T=4     batch config
 */
#include "bench.h"
#include "wubu_model.h"
#include "wubu_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Scratch buffer types (mirror bench_e2e.c)
typedef struct {
    float *d_qkv; float *d_z; float *d_beta; float *d_alpha;
    float *d_beta_sig; float *d_alpha_bi; float *d_gate;
    float *d_conv_input; float *d_conv_out;
    float *d_q_conv; float *d_k_conv; float *d_v_conv;
    float *d_q_norm; float *d_k_norm;
    float *d_delta_out; float *d_z_silu;
} gpu_ssm_scratch;

typedef struct {
    float *d_Q_full; float *d_K; float *d_V; float *d_scratch;
} gpu_gqa_scratch;

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1]
        : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *corpus_path = argc > 2 ? argv[2] : "data/train_data.bin";
    int n_steps = argc > 3 ? atoi(argv[3]) : 10;
    float lr = 0.001f;
    if (getenv("LR")) lr = atof(getenv("LR"));
    const char *embed_path = "data/qwen36_embeddings_c.bin";
    
    int B = 1, T = 4, N = B * T;
    
    printf("=== WuBuText AI — GPU Training ===\n");
    printf("Model: %s | Steps: %d | LR: %.4f | B=%d T=%d\n",
           model_path, n_steps, lr, B, T);
    fflush(stdout);
    
    // Load tokenizer
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, model_path)) return 1;
    int vocab_size = tok.vocab_size;
    printf("Vocab: %d\n", vocab_size);
    
    // Load model
    wubu_model_t model;
    if (!wubu_model_init(&model, model_path)) return 1;
    printf("Model: %d layers\n", model.n_layers);
    
    // Load training data
    FILE *f = fopen(corpus_path, "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    int total_tokens = (int)(ftell(f) / sizeof(int));
    fseek(f, 0, SEEK_SET);
    int *tokens = (int *)malloc(total_tokens * sizeof(int));
    fread(tokens, sizeof(int), total_tokens, f);
    fclose(f);
    printf("Corpus: %d tokens\n", total_tokens);
    
    // Load output.weight as mutable float
    float *output_weight = (float *)malloc(D_MODEL * vocab_size * sizeof(float));
    {
        gguf_ctx *ctx = gguf_open(model_path);
        gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
        gguf_read_tensor_f32(ctx, t, output_weight, D_MODEL * vocab_size);
        gguf_close(ctx);
    }
    printf("Output weight: loaded\n");
    
    // ========== GPU Init ==========
    cublasHandle_t cublas_h;
    cudaStream_t stream;
    if (!wubu_cuda_init(&cublas_h, &stream)) {
        fprintf(stderr, "CUDA init failed\n");
        return 1;
    }
    printf("CUDA: initialized\n");
    
    // Open GGUF for per-layer weight loading (bench_e2e pattern)
    gguf_ctx *ctx = gguf_open(model_path);
    if (!ctx) { fprintf(stderr, "GGUF open failed\n"); return 1; }
    
    // Allocate GPU buffers
    float *d_x = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
    float *d_out = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
    float *d_norm = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
    float *d_norm_weight = wubu_cuda_alloc(D_MODEL * sizeof(float));
    
    // SSM states
    float **d_ssm_states = calloc(model.n_layers, sizeof(float *));
    float **d_conv_states = calloc(model.n_layers, sizeof(float *));
    for (int l = 0; l < model.n_layers; l++) {
        if (model.layers[l].is_ssm) {
            d_ssm_states[l] = wubu_cuda_alloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
            d_conv_states[l] = wubu_cuda_alloc((CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
            cudaMemsetAsync(d_ssm_states[l], 0, SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float), stream);
            cudaMemsetAsync(d_conv_states[l], 0, (CONV_KERNEL - 1) * CONV_DIM * sizeof(float), stream);
        }
    }
    
    // Scratch buffers
    gpu_ssm_scratch ssm_scr;
    gpu_gqa_scratch gqa_scr;
    int qkv_dim = KEY_DIM * 2 + VALUE_DIM;
    int q_dim_x2 = GQA_Q_HEADS * GQA_HEAD_DIM * 2;
    int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
    
    ssm_scr.d_qkv        = wubu_cuda_alloc(N * qkv_dim * sizeof(float));
    ssm_scr.d_z          = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
    ssm_scr.d_beta       = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    ssm_scr.d_alpha      = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    ssm_scr.d_beta_sig   = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    ssm_scr.d_alpha_bi   = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    ssm_scr.d_gate       = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    ssm_scr.d_conv_input = wubu_cuda_alloc(B * (T + CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
    ssm_scr.d_conv_out   = wubu_cuda_alloc(N * CONV_DIM * sizeof(float));
    ssm_scr.d_q_conv     = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    ssm_scr.d_k_conv     = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    ssm_scr.d_v_conv     = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
    ssm_scr.d_q_norm     = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    ssm_scr.d_k_norm     = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    ssm_scr.d_delta_out  = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
    ssm_scr.d_z_silu     = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
    
    gqa_scr.d_Q_full  = wubu_cuda_alloc(N * q_dim_x2 * sizeof(float));
    gqa_scr.d_K       = wubu_cuda_alloc(N * kv_dim * sizeof(float));
    gqa_scr.d_V       = wubu_cuda_alloc(N * kv_dim * sizeof(float));
    gqa_scr.d_scratch = wubu_cuda_alloc(N * GQA_Q_HEADS * GQA_HEAD_DIM * sizeof(float));
    
    cudaStreamSynchronize(stream);
    
    // CPU buffers
    float *embd = (float *)malloc(N * D_MODEL * sizeof(float));
    float *hidden = (float *)malloc(N * D_MODEL * sizeof(float));
    float *logits = (float *)malloc(N * vocab_size * sizeof(float));
    float *dlogits = (float *)malloc(N * vocab_size * sizeof(float));
    float *dW = (float *)malloc(D_MODEL * vocab_size * sizeof(float));
    float *norm_weight_buf = (float *)malloc(D_MODEL * sizeof(float));
    
    printf("\n=== Training: %d steps ===\n\n", n_steps);
    
    double total_time = 0.0;
    for (int step = 0; step < n_steps; step++) {
        int start_idx = (step * N) % (total_tokens - N - 1);
        
        // Load embeddings from file
        f = fopen(embed_path, "rb");
        for (int i = 0; i < N; i++) {
            int id = tokens[start_idx + i];
            if (id < 0 || id >= model.vocab_size) id = 0;
            fseek(f, id * D_MODEL * sizeof(float), SEEK_SET);
            fread(embd + i * D_MODEL, sizeof(float), D_MODEL, f);
        }
        fclose(f);
        
        // Targets
        int targets[N];
        for (int i = 0; i < N - 1; i++) targets[i] = tokens[start_idx + i + 1];
        targets[N - 1] = 0;
        
        double t0 = now_sec();
        
        // === GPU Forward Pass ===
        // Copy embeddings to GPU
        cudaMemcpyAsync(d_x, embd, N * D_MODEL * sizeof(float), cudaMemcpyHostToDevice, stream);
        
        // Run layers on GPU — no pointer swap, d_cur is persistent residual
        // Each layer: norm(d_cur) → attention → saxpy(d_cur += attn_out)
        float *d_cur = d_x;       // starts with embeddings, accumulates residuals
        float *d_norm_p = d_norm; // scratch for norm output
        
        for (int l = 0; l < model.n_layers; l++) {
            // Pre-attention RMSNorm on GPU: norm(d_cur, w) → d_norm_p
            cudaMemcpyAsync(d_norm_weight, model.layers[l].attn_norm_weight,
                          D_MODEL * sizeof(float), cudaMemcpyHostToDevice, stream);
            wubu_cuda_rms_norm(B, T, D_MODEL, d_cur, d_norm_weight, 1e-6f, d_norm_p, stream);
            
            // Attention on GPU: reads from d_norm_p, writes to d_out
            if (model.layers[l].is_ssm) {
                gpu_ssm_weights w;
                gpu_load_ssm_layer(ctx, l, &w, stream);
                gpu_ssm_forward(cublas_h, stream, d_norm_p, B, T,
                    w.d_attn_qkv, w.d_attn_gate,
                    w.d_ssm_beta, w.d_ssm_alpha,
                    w.d_ssm_dt_bias, w.d_ssm_a,
                    w.d_ssm_conv1d, w.d_ssm_norm, w.d_ssm_out,
                    d_ssm_states[l], d_conv_states[l],
                    d_out,
                    ssm_scr.d_qkv, ssm_scr.d_z,
                    ssm_scr.d_beta, ssm_scr.d_alpha,
                    ssm_scr.d_beta_sig, ssm_scr.d_alpha_bi, ssm_scr.d_gate,
                    ssm_scr.d_conv_input, ssm_scr.d_conv_out,
                    ssm_scr.d_q_conv, ssm_scr.d_k_conv, ssm_scr.d_v_conv,
                    ssm_scr.d_q_norm, ssm_scr.d_k_norm,
                    ssm_scr.d_delta_out, ssm_scr.d_z_silu);
                gpu_free_ssm_weights(&w);
            } else {
                gpu_gqa_weights w;
                gpu_load_gqa_layer(ctx, l, &w, stream);
                gpu_gqa_forward(cublas_h, stream, d_norm_p, B, T,
                    w.d_attn_q, w.d_attn_k, w.d_attn_v,
                    w.d_attn_out_w, w.d_q_norm_w, w.d_k_norm_w,
                    d_out,
                    gqa_scr.d_Q_full, gqa_scr.d_K, gqa_scr.d_V, gqa_scr.d_scratch);
                gpu_free_gqa_weights(&w);
            }
            
            // Residual: d_cur = d_cur + d_out (saxpy)
            float alpha = 1.0f;
            cublasSaxpy(cublas_h, N * D_MODEL, &alpha, d_out, 1, d_cur, 1);
            cudaStreamSynchronize(stream);
        }
        
        // Copy final hidden states to CPU (d_cur has final residual)
        cudaMemcpy(hidden, d_cur, N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(stream);
        
        // === CPU: Output projection + Loss + Gradient ===
        for (int i = 0; i < N; i++) {
            const float *h = hidden + i * D_MODEL;
            float *log_i = logits + i * vocab_size;
            for (int j = 0; j < vocab_size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < D_MODEL; k++)
                    sum += h[k] * output_weight[j * D_MODEL + k];
                log_i[j] = sum;
            }
        }
        
        // CE loss + gradient
        float loss = 0.0f;
        for (int i = 0; i < N; i++) {
            float max_l = logits[i * vocab_size];
            for (int j = 1; j < vocab_size; j++)
                if (logits[i * vocab_size + j] > max_l) max_l = logits[i * vocab_size + j];
            float sum_exp = 0.0f;
            for (int j = 0; j < vocab_size; j++) {
                float e = expf(logits[i * vocab_size + j] - max_l);
                dlogits[i * vocab_size + j] = e;
                sum_exp += e;
            }
            float inv_sum = 1.0f / (sum_exp + 1e-30f);
            for (int j = 0; j < vocab_size; j++) {
                float soft = dlogits[i * vocab_size + j] * inv_sum;
                dlogits[i * vocab_size + j] = soft - (j == targets[i] ? 1.0f : 0.0f);
            }
            float soft_t = expf(logits[i * vocab_size + targets[i]] - max_l) * inv_sum;
            loss += -logf(soft_t + 1e-30f);
        }
        loss /= N;
        
        // Gradient w.r.t. output.weight
        memset(dW, 0, D_MODEL * vocab_size * sizeof(float));
        for (int j = 0; j < vocab_size; j++) {
            for (int k = 0; k < D_MODEL; k++) {
                float sum = 0.0f;
                for (int i = 0; i < N; i++)
                    sum += hidden[i * D_MODEL + k] * dlogits[i * vocab_size + j];
                dW[j * D_MODEL + k] = sum / N;
            }
        }
        
        // SGD with gradient clipping
        float max_g = 0.0f;
        for (int64_t idx = 0; idx < (int64_t)D_MODEL * vocab_size; idx++)
            if (fabsf(dW[idx]) > max_g) max_g = fabsf(dW[idx]);
        float clip = max_g > 1.0f ? 1.0f / max_g : 1.0f;
        for (int64_t idx = 0; idx < (int64_t)D_MODEL * vocab_size; idx++)
            output_weight[idx] -= lr * dW[idx] * clip;
        
        double step_time = now_sec() - t0;
        total_time += step_time;
        
        printf("Step %3d: loss=%.4f (%.3fs, %.1f tok/s)\n",
               step + 1, loss, step_time, N / step_time);
        fflush(stdout);
    }
    
    printf("\n=== RESULTS ===\n");
    printf("Avg time/step: %.3fs (%.1f tok/s)\n",
           total_time / n_steps, N / (total_time / n_steps));
    printf("Output weight: trained via SGD (lr=%.4f)\n", lr);
    
    // Cleanup
    for (int l = 0; l < model.n_layers; l++) {
        wubu_cuda_free(d_ssm_states[l]);
        wubu_cuda_free(d_conv_states[l]);
    }
    free(d_ssm_states); free(d_conv_states);
    wubu_cuda_free(d_x); wubu_cuda_free(d_out); wubu_cuda_free(d_norm);
    wubu_cuda_free(d_norm_weight);
    wubu_cuda_free(ssm_scr.d_qkv); wubu_cuda_free(ssm_scr.d_z);
    wubu_cuda_free(ssm_scr.d_beta); wubu_cuda_free(ssm_scr.d_alpha);
    wubu_cuda_free(ssm_scr.d_beta_sig); wubu_cuda_free(ssm_scr.d_alpha_bi);
    wubu_cuda_free(ssm_scr.d_gate); wubu_cuda_free(ssm_scr.d_conv_input);
    wubu_cuda_free(ssm_scr.d_conv_out); wubu_cuda_free(ssm_scr.d_q_conv);
    wubu_cuda_free(ssm_scr.d_k_conv); wubu_cuda_free(ssm_scr.d_v_conv);
    wubu_cuda_free(ssm_scr.d_q_norm); wubu_cuda_free(ssm_scr.d_k_norm);
    wubu_cuda_free(ssm_scr.d_delta_out); wubu_cuda_free(ssm_scr.d_z_silu);
    wubu_cuda_free(gqa_scr.d_Q_full); wubu_cuda_free(gqa_scr.d_K);
    wubu_cuda_free(gqa_scr.d_V); wubu_cuda_free(gqa_scr.d_scratch);
    wubu_cuda_destroy(cublas_h, stream);
    gguf_close(ctx);
    free(tokens); free(embd); free(hidden); free(logits);
    free(dlogits); free(dW); free(output_weight); free(norm_weight_buf);
    wubu_tokenizer_free(&tok);
    wubu_model_free(&model);
    return 0;
}
