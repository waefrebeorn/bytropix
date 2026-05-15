/**
 * train_gpu.c — Phase 3.5: GPU-Accelerated Training
 *
 * Hybrid: GPU for attention/SSM forward, CPU for norms + output projection.
 * Full internal weight gradient accumulation + SGD via Q-learner LR.
 *
 * Usage: ./train_gpu [model.gguf] [corpus.bin] [steps]
 *   LR=0.001    learning rate (overridden by Q-learner)
 *   B=1 T=4     batch config
 */
#include "bench.h"
#include "wubu_model.h"
#include "wubu_tokenizer.h"
#include "qlearner.h"
#include "gguf_reader.h"
#include "rsgd.h"
#include "wubu_tst.h"
#include "wubu_poincare_gqa.h"
#include "wubu_nested_ssm.h"
#include "wubu_moe_hyperbolic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

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

typedef struct { float *ptr; long long sz; } weight_grad_t;

// Per-layer MoE cached data for lazy dequant (forward→backward)
typedef struct {
    int n_unique;
    int unique_ids[32];
    float *deq_gate_inp;
    float *gate_shexp, *up_shexp, *down_shexp;
    float *expert_gate[32], *expert_up[32], *expert_down[32];
} lazy_moe_cache_t;

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1]
        : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *corpus_path = argc > 2 ? argv[2] : "data/train_data.bin";
    int n_steps = argc > 3 ? atoi(argv[3]) : 10;
    float lr = 0.001f;
    if (getenv("LR")) lr = atof(getenv("LR"));
    float poincare_R = 0.0f;
    if (getenv("POINCARE_R")) poincare_R = atof(getenv("POINCARE_R"));
    const char *embed_path = "data/qwen36_embeddings_c.bin.raw";
    
    // Integration flags
    int tst_enabled = getenv("TST") ? atoi(getenv("TST")) : 0;
    int rsgd_enabled = getenv("RSGD") ? atoi(getenv("RSGD")) : 0;
    int pga_enabled = getenv("PGA") ? atoi(getenv("PGA")) : 0;
    int nested_ssm_enabled = getenv("NESTED_SSM") ? atoi(getenv("NESTED_SSM")) : 0;
    int nested_moe_enabled = getenv("NESTED_MOE") ? atoi(getenv("NESTED_MOE")) : 0;
    int tst_bag_size = tst_enabled ? 8 : 0;
    
    int B = 1, T = getenv("TST") ? 16 : 4, N = B * T;
    (void)B;
    
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    
    printf("=== WuBuText AI — GPU Training ===\n");
    printf("Model: %s | Steps: %d | LR: %.6f | B=%d T=%d\n",
           model_path, n_steps, lr, B, T);
    if (poincare_R > 0.0f)
        printf("Poincaré: R=%.4f (hyperbolic SSM recurrence)\n", poincare_R);
    fflush(stdout);
    
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, model_path)) return 1;
    int vocab_size = tok.vocab_size;
    printf("Vocab: %d\n", vocab_size);
    
    wubu_model_t model;
    if (!wubu_model_init(&model, model_path)) return 1;
    printf("Model: %d layers\n", model.n_layers);
    
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
    
    // Q-learner init
    qlearner_t ql;
    qlearner_init(&ql);
    printf("Q-learner: init (lr=%.6f)\n", ql.lr);
    
    // GPU Init
    cublasHandle_t cublas_h;
    cudaStream_t stream;
    if (!wubu_cuda_init(&cublas_h, &stream)) return 1;
    printf("CUDA: initialized\n");
    
    // Pre-load all layer weights to GPU
    printf("Pre-loading all layer weights to GPU...\n");
    gguf_ctx *ctx = gguf_open(model_path);
    if (!ctx) return 1;
    
    gpu_ssm_weights *ssm_gpu_weights = calloc(model.n_layers, sizeof(gpu_ssm_weights));
    gpu_gqa_weights *gqa_gpu_weights = calloc(model.n_layers, sizeof(gpu_gqa_weights));
    
    double t_load = now_sec();
    for (int l = 0; l < model.n_layers; l++) {
        if (model.layers[l].is_ssm) {
            if (!gpu_load_ssm_layer(ctx, l, &ssm_gpu_weights[l], stream)) return 1;
        } else {
            if (!gpu_load_gqa_layer(ctx, l, &gqa_gpu_weights[l], stream)) return 1;
        }
    }
    cudaStreamSynchronize(stream);
    gguf_close(ctx);
    printf("  All %d layers loaded in %.2fs\n", model.n_layers, now_sec() - t_load);
    
    // GPU buffers
    float *d_x = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
    float *d_out = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
    float *d_norm = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
    float *d_norm_weight = wubu_cuda_alloc(D_MODEL * sizeof(float));
    float *d_poincare_norms = poincare_R > 0.0f ? wubu_cuda_alloc(N * sizeof(float)) : NULL;
    
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
    int qkv_dim = KEY_DIM * 2 + VALUE_DIM;
    int q_dim_x2 = GQA_Q_HEADS * GQA_HEAD_DIM * 2;
    int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
    int gqa_q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
    int state_sz = SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    
    gpu_ssm_scratch ssm_scr;
    gpu_gqa_scratch gqa_scr;
    // ... [scratch allocs same as before - omitted for brevity, see original]
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
    gqa_scr.d_scratch = wubu_cuda_alloc(N * gqa_q_dim * sizeof(float));
    
    // Per-layer trajectory buffers (same as original)
    float **d_states_t = calloc(model.n_layers, sizeof(float *));
    float **cpu_ssm_scr = calloc(model.n_layers, sizeof(float *));
    for (int l = 0; l < model.n_layers; l++) {
        if (model.layers[l].is_ssm) {
            d_states_t[l] = wubu_cuda_alloc((T+1) * state_sz * sizeof(float));
            int layer_bytes = N * (qkv_dim + VALUE_DIM + 4*DT_RANK + CONV_DIM + 2*KEY_DIM + VALUE_DIM + 2*KEY_DIM + VALUE_DIM + VALUE_DIM) * sizeof(float)
                + (T+1) * state_sz * sizeof(float)
                + (CONV_KERNEL-1) * CONV_DIM * sizeof(float);
            cpu_ssm_scr[l] = (float *)malloc(layer_bytes);
        }
    }
    
    float **d_gqa_q_norm_save = calloc(model.n_layers, sizeof(float *));
    float **d_gqa_k_raw_save = calloc(model.n_layers, sizeof(float *));
    float **cpu_gqa_scr = calloc(model.n_layers, sizeof(float *));
    for (int l = 0; l < model.n_layers; l++) {
        if (!model.layers[l].is_ssm) {
            d_gqa_q_norm_save[l] = wubu_cuda_alloc(N * gqa_q_dim * sizeof(float));
            d_gqa_k_raw_save[l] = wubu_cuda_alloc(N * kv_dim * sizeof(float));
            int bytes = N * (q_dim_x2 + kv_dim + gqa_q_dim + kv_dim + kv_dim + gqa_q_dim) * sizeof(float);
            cpu_gqa_scr[l] = (float *)malloc(bytes);
        }
    }
    
    cudaStreamSynchronize(stream);
    
    // CPU buffers
    float *embd = (float *)malloc(N * D_MODEL * sizeof(float));
    float *hidden = (float *)malloc(N * D_MODEL * sizeof(float));
    float *logits = (float *)malloc(N * vocab_size * sizeof(float));
    float *dlogits = (float *)malloc(N * vocab_size * sizeof(float));
    float *dW = (float *)malloc(D_MODEL * vocab_size * sizeof(float));
    float *norm_weight_buf = (float *)malloc(D_MODEL * sizeof(float));
    
    float *hidden_per_layer = NULL, *saved_normed = NULL, *saved_attn_out = NULL;
    float *saved_normed2 = NULL, *saved_ffn_out = NULL;
    if (model.n_layers > 0) {
        int total_sz = model.n_layers * N * D_MODEL;
        hidden_per_layer = (float *)malloc(total_sz * sizeof(float));
        saved_normed = (float *)calloc(total_sz, sizeof(float));
        saved_attn_out = (float *)calloc(total_sz, sizeof(float));
        saved_normed2 = (float *)calloc(total_sz, sizeof(float));
        saved_ffn_out = (float *)calloc(total_sz, sizeof(float));
    }
    
    printf("\n=== Training: %d steps ===\n\n", n_steps);

    // Reopen GGUF for MoE lazy loading during training
    gguf_ctx *gguf_moe = gguf_open(model_path);
    if (!gguf_moe) { fprintf(stderr, "Failed to reopen GGUF for MoE\n"); return 1; }
    // Buffer entire GGUF in RAM for fast MoE weight access (no SSD seeks)
    gguf_buffer_data(gguf_moe);
    
    // Pre-compute quantized MoE tensor pointers per layer
    typedef struct {
        const uint8_t *q_gate_inp, *q_gate_exps, *q_up_exps, *q_down_exps;
        const uint8_t *q_gate_shexp, *q_up_shexp, *q_down_shexp;
        int ty_gi, ty_ge, ty_gs;
        int64_t expert_raw, expert_raw_down;
    } moe_qdata_t;
    moe_qdata_t *moe_qdata = calloc(model.n_layers, sizeof(moe_qdata_t));
    lazy_moe_cache_t *moe_cache = calloc(model.n_layers, sizeof(lazy_moe_cache_t));
    int moe_enabled = 1;
    if (getenv("NO_MOE")) moe_enabled = 0;
    
    char mname[256];
    for (int l = 0; l < model.n_layers; l++) {
        moe_qdata_t *mq = &moe_qdata[l];
        gguf_tensor_info *t;
        snprintf(mname, sizeof(mname), "blk.%d.ffn_gate_inp.weight", l);
        t = gguf_find_tensor(gguf_moe, mname);
        if (t) { mq->ty_gi = t->ggml_type; mq->q_gate_inp = (const uint8_t *)gguf_moe->data_blob + t->data_offset; }
        snprintf(mname, sizeof(mname), "blk.%d.ffn_gate_exps.weight", l);
        t = gguf_find_tensor(gguf_moe, mname);
        if (t) { mq->ty_ge = t->ggml_type; mq->q_gate_exps = (const uint8_t *)gguf_moe->data_blob + t->data_offset; }
        snprintf(mname, sizeof(mname), "blk.%d.ffn_up_exps.weight", l);
        t = gguf_find_tensor(gguf_moe, mname);
        if (t) mq->q_up_exps = (const uint8_t *)gguf_moe->data_blob + t->data_offset;
        snprintf(mname, sizeof(mname), "blk.%d.ffn_down_exps.weight", l);
        t = gguf_find_tensor(gguf_moe, mname);
        if (t) mq->q_down_exps = (const uint8_t *)gguf_moe->data_blob + t->data_offset;
        snprintf(mname, sizeof(mname), "blk.%d.ffn_gate_shexp.weight", l);
        t = gguf_find_tensor(gguf_moe, mname);
        if (t) { mq->ty_gs = t->ggml_type; mq->q_gate_shexp = (const uint8_t *)gguf_moe->data_blob + t->data_offset; }
        snprintf(mname, sizeof(mname), "blk.%d.ffn_up_shexp.weight", l);
        t = gguf_find_tensor(gguf_moe, mname);
        if (t) mq->q_up_shexp = (const uint8_t *)gguf_moe->data_blob + t->data_offset;
        snprintf(mname, sizeof(mname), "blk.%d.ffn_down_shexp.weight", l);
        t = gguf_find_tensor(gguf_moe, mname);
        if (t) mq->q_down_shexp = (const uint8_t *)gguf_moe->data_blob + t->data_offset;
        mq->expert_raw = gguf_raw_size(mq->ty_ge, (int64_t)D_MODEL * D_FF);
        mq->expert_raw_down = gguf_raw_size(mq->ty_ge, (int64_t)D_FF * D_MODEL);
    }
    if (moe_enabled) printf("  MoE: lazy (top-%d/%d experts per layer)\n", N_ACTIVE_EXPTS, N_EXPERTS);
    else printf("  MoE: disabled (identity)\n");
    
    double total_time = 0.0;
    for (int step = 0; step < n_steps; step++) {
        int start_idx = (step * N) % (total_tokens - N - 1);
        
        // Load embeddings
        f = fopen(embed_path, "rb");
        for (int i = 0; i < N; i++) {
            int id = tokens[start_idx + i];
            if (id < 0 || id >= model.vocab_size) id = 0;
            fseek(f, id * D_MODEL * sizeof(float), SEEK_SET);
            fread(embd + i * D_MODEL, sizeof(float), D_MODEL, f);
        }
        // === TST: Bag embeddings for superposition phase ===
        int T_eff = T, N_eff = N;
        int tst_targets_buf[/*max bags*/ 64 * 8]; // [N/s, s]
        int n_tst_bags = 0;
        float bagged_embd[/*N/8 max*/ 64 * D_MODEL];
        
        if (tst_enabled && step % 4 == 0) { // 25% superposition steps
            // Bag embeddings
            int s = tst_bag_size;
            int T_bagged = T / s;
            int N_bagged = N / s;
            for (int b = 0; b < B; b++) {
                for (int t = 0; t < T_bagged; t++) {
                    float *out = bagged_embd + (b * T_bagged + t) * D_MODEL;
                    memset(out, 0, D_MODEL * sizeof(float));
                    for (int k = 0; k < s; k++) {
                        const float *in = embd + (b * T + t * s + k) * D_MODEL;
                        for (int d = 0; d < D_MODEL; d++) out[d] += in[d] / s;
                    }
                }
            }
            // Targets: shift left by s-1, create bags
            int shifted[64];
            for (int i = 0; i < N - s + 1; i++) shifted[i] = tokens[start_idx + i + s - 1];
            n_tst_bags = tst_prepare_targets(shifted, tst_targets_buf, B, T - s + 1, s);
            T_eff = T_bagged; N_eff = N_bagged;
        }
        
        fclose(f);
        
        int targets[N];
        for (int i = 0; i < N - 1; i++) targets[i] = tokens[start_idx + i + 1];
        targets[N - 1] = 0;
        
        double t0 = now_sec();
        
        // === GPU Forward Pass ===
        const float *fwd_embd = (tst_enabled && step % 4 == 0) ? bagged_embd : embd;
        int use_tst_step = (tst_enabled && step % 4 == 0);
        int fwd_B = B;
        (void)fwd_B;
        int fwd_T = use_tst_step ? T_eff : T;
        int fwd_N = fwd_B * fwd_T;
        
        cudaMemcpyAsync(d_x, fwd_embd, fwd_N * D_MODEL * sizeof(float), cudaMemcpyHostToDevice, stream);
        
        if (poincare_R > 0.0f) {
            wubu_cuda_norm(d_x, d_poincare_norms, fwd_N, D_MODEL, stream);
            wubu_cuda_exp_map(d_x, d_poincare_norms, poincare_R, d_x, fwd_N, D_MODEL, stream);
        }
        
        float *d_cur = d_x;
        float *d_norm_p = d_norm;
        
        for (int l = 0; l < model.n_layers; l++) {
            cudaMemcpyAsync(d_norm_weight, model.layers[l].attn_norm_weight,
                          D_MODEL * sizeof(float), cudaMemcpyHostToDevice, stream);
            wubu_cuda_rms_norm(B, T, D_MODEL, d_cur, d_norm_weight, 1e-6f, d_norm_p, stream);
            
            cudaMemcpy(saved_normed + l * N * D_MODEL, d_norm_p,
                      N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);
            
            if (model.layers[l].is_ssm) {
                if (poincare_R > 0.0f) {
                    gpu_poincare_ssm_forward(cublas_h, stream, d_norm_p, B, T,
                        ssm_gpu_weights[l].d_attn_qkv, ssm_gpu_weights[l].d_attn_gate,
                        ssm_gpu_weights[l].d_ssm_beta, ssm_gpu_weights[l].d_ssm_alpha,
                        ssm_gpu_weights[l].d_ssm_dt_bias, ssm_gpu_weights[l].d_ssm_a,
                        ssm_gpu_weights[l].d_ssm_conv1d, ssm_gpu_weights[l].d_ssm_norm, ssm_gpu_weights[l].d_ssm_out,
                        d_ssm_states[l], d_conv_states[l],
                        d_out,
                        ssm_scr.d_qkv, ssm_scr.d_z,
                        ssm_scr.d_beta, ssm_scr.d_alpha,
                        ssm_scr.d_beta_sig, ssm_scr.d_alpha_bi, ssm_scr.d_gate,
                        ssm_scr.d_conv_input, ssm_scr.d_conv_out,
                        ssm_scr.d_q_conv, ssm_scr.d_k_conv, ssm_scr.d_v_conv,
                        ssm_scr.d_q_norm, ssm_scr.d_k_norm,
                        ssm_scr.d_delta_out, ssm_scr.d_z_silu,
                        poincare_R);
                } else {
                    gpu_ssm_forward_save(cublas_h, stream, d_norm_p, B, T,
                        ssm_gpu_weights[l].d_attn_qkv, ssm_gpu_weights[l].d_attn_gate,
                        ssm_gpu_weights[l].d_ssm_beta, ssm_gpu_weights[l].d_ssm_alpha,
                        ssm_gpu_weights[l].d_ssm_dt_bias, ssm_gpu_weights[l].d_ssm_a,
                        ssm_gpu_weights[l].d_ssm_conv1d, ssm_gpu_weights[l].d_ssm_norm, ssm_gpu_weights[l].d_ssm_out,
                        d_ssm_states[l], d_conv_states[l],
                        d_states_t ? d_states_t[l] : NULL,
                        d_out,
                        ssm_scr.d_qkv, ssm_scr.d_z,
                        ssm_scr.d_beta, ssm_scr.d_alpha,
                        ssm_scr.d_beta_sig, ssm_scr.d_alpha_bi, ssm_scr.d_gate,
                        ssm_scr.d_conv_input, ssm_scr.d_conv_out,
                        ssm_scr.d_q_conv, ssm_scr.d_k_conv, ssm_scr.d_v_conv,
                        ssm_scr.d_q_norm, ssm_scr.d_k_norm,
                        ssm_scr.d_delta_out, ssm_scr.d_z_silu);
                    
                    if (cpu_ssm_scr && cpu_ssm_scr[l]) {
                        float *p = cpu_ssm_scr[l];
                        cudaMemcpy(p, ssm_scr.d_qkv, N * qkv_dim * sizeof(float), cudaMemcpyDeviceToHost); p += N * qkv_dim;
                        cudaMemcpy(p, ssm_scr.d_z, N * VALUE_DIM * sizeof(float), cudaMemcpyDeviceToHost); p += N * VALUE_DIM;
                        cudaMemcpy(p, ssm_scr.d_beta, N * DT_RANK * sizeof(float), cudaMemcpyDeviceToHost); p += N * DT_RANK;
                        cudaMemcpy(p, ssm_scr.d_alpha, N * DT_RANK * sizeof(float), cudaMemcpyDeviceToHost); p += N * DT_RANK;
                        cudaMemcpy(p, ssm_scr.d_beta_sig, N * DT_RANK * sizeof(float), cudaMemcpyDeviceToHost); p += N * DT_RANK;
                        cudaMemcpy(p, ssm_scr.d_gate, N * DT_RANK * sizeof(float), cudaMemcpyDeviceToHost); p += N * DT_RANK;
                        cudaMemcpy(p, ssm_scr.d_conv_out, N * CONV_DIM * sizeof(float), cudaMemcpyDeviceToHost); p += N * CONV_DIM;
                        cudaMemcpy(p, ssm_scr.d_q_conv, N * KEY_DIM * sizeof(float), cudaMemcpyDeviceToHost); p += N * KEY_DIM;
                        cudaMemcpy(p, ssm_scr.d_k_conv, N * KEY_DIM * sizeof(float), cudaMemcpyDeviceToHost); p += N * KEY_DIM;
                        cudaMemcpy(p, ssm_scr.d_v_conv, N * VALUE_DIM * sizeof(float), cudaMemcpyDeviceToHost); p += N * VALUE_DIM;
                        cudaMemcpy(p, ssm_scr.d_q_norm, N * KEY_DIM * sizeof(float), cudaMemcpyDeviceToHost); p += N * KEY_DIM;
                        cudaMemcpy(p, ssm_scr.d_k_norm, N * KEY_DIM * sizeof(float), cudaMemcpyDeviceToHost); p += N * KEY_DIM;
                        cudaMemcpy(p, ssm_scr.d_delta_out, N * VALUE_DIM * sizeof(float), cudaMemcpyDeviceToHost); p += N * VALUE_DIM;
                        cudaMemcpy(p, ssm_scr.d_z_silu, N * VALUE_DIM * sizeof(float), cudaMemcpyDeviceToHost); p += N * VALUE_DIM;
                        cudaMemcpy(p, d_states_t[l], (T+1) * state_sz * sizeof(float), cudaMemcpyDeviceToHost); p += (T+1) * state_sz;
                        cudaMemcpy(p, d_conv_states[l], (CONV_KERNEL-1) * CONV_DIM * sizeof(float), cudaMemcpyDeviceToHost);
                    }
                }
            } else {
                gpu_gqa_forward_save(cublas_h, stream, d_norm_p, B, T,
                    gqa_gpu_weights[l].d_attn_q, gqa_gpu_weights[l].d_attn_k, gqa_gpu_weights[l].d_attn_v,
                    gqa_gpu_weights[l].d_attn_out_w, gqa_gpu_weights[l].d_q_norm_w, gqa_gpu_weights[l].d_k_norm_w,
                    d_out,
                    gqa_scr.d_Q_full, gqa_scr.d_K, gqa_scr.d_V, gqa_scr.d_scratch,
                    d_gqa_q_norm_save[l], d_gqa_k_raw_save[l],
                    gqa_scr.d_K, gqa_scr.d_scratch);
                
                if (cpu_gqa_scr && cpu_gqa_scr[l]) {
                    float *p = cpu_gqa_scr[l];
                    cudaMemcpy(p, gqa_scr.d_Q_full, N * q_dim_x2 * sizeof(float), cudaMemcpyDeviceToHost); p += N * q_dim_x2;
                    cudaMemcpy(p, d_gqa_k_raw_save[l], N * kv_dim * sizeof(float), cudaMemcpyDeviceToHost); p += N * kv_dim;
                    cudaMemcpy(p, d_gqa_q_norm_save[l], N * gqa_q_dim * sizeof(float), cudaMemcpyDeviceToHost); p += N * gqa_q_dim;
                    cudaMemcpy(p, gqa_scr.d_K, N * kv_dim * sizeof(float), cudaMemcpyDeviceToHost); p += N * kv_dim;
                    cudaMemcpy(p, gqa_scr.d_V, N * kv_dim * sizeof(float), cudaMemcpyDeviceToHost); p += N * kv_dim;
                    cudaMemcpy(p, gqa_scr.d_scratch, N * gqa_q_dim * sizeof(float), cudaMemcpyDeviceToHost); p += N * gqa_q_dim;
                    cudaStreamSynchronize(stream);
                }
            }
            
            cudaMemcpy(saved_attn_out + l * N * D_MODEL, d_out,
                      N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);
            
            float alpha = 1.0f;
            cublasSaxpy(cublas_h, N * D_MODEL, &alpha, d_out, 1, d_cur, 1);
            
            cudaMemcpy(hidden_per_layer + l * N * D_MODEL, d_cur,
                      N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);
            
            cudaMemcpyAsync(d_norm_weight, model.layers[l].post_attn_norm_weight,
                          D_MODEL * sizeof(float), cudaMemcpyHostToDevice, stream);
            wubu_cuda_rms_norm(B, T, D_MODEL, d_cur, d_norm_weight, 1e-6f, d_norm_p, stream);
            
            cudaMemcpy(saved_normed2 + l * N * D_MODEL, d_norm_p,
                      N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);
            // === Lazy MoE forward (replaces identity) ===
            if (moe_enabled && moe_qdata[l].q_gate_exps) {
                moe_qdata_t *mq = &moe_qdata[l];
                lazy_moe_cache_t *mc = &moe_cache[l];
                const float *n2 = saved_normed2 + l * N * D_MODEL;
                float *ffn_out = saved_ffn_out + l * N * D_MODEL;
                
                // 1. Dequantize router
                if (!mc->deq_gate_inp)
                    mc->deq_gate_inp = (float *)malloc(D_MODEL * N_EXPERTS * sizeof(float));
                gguf_dequantize(mq->q_gate_inp, mq->ty_gi, D_MODEL * N_EXPERTS, mc->deq_gate_inp);
                
                // 2. Route: compute scores
                float *scores = (float *)malloc(N * N_EXPERTS * sizeof(float));
                for (int s = 0; s < N; s++) {
                    for (int e = 0; e < N_EXPERTS; e++) {
                        double sum = 0.0;
                        for (int k = 0; k < D_MODEL; k++)
                            sum += (double)n2[s*D_MODEL+k] * (double)mc->deq_gate_inp[e*D_MODEL+k];
                        scores[s*N_EXPERTS+e] = (float)sum;
                    }
                }
                
                // 3. Softmax → top-k → unique expert IDs
                int topk_idx[4*8]; float topk_wt[4*8];
                mc->n_unique = 0;
                for (int s = 0; s < N; s++) {
                    float *sc = scores + s * N_EXPERTS;
                    float mx = sc[0]; for (int e=1; e<N_EXPERTS; e++) if(sc[e]>mx) mx=sc[e];
                    float se = 0; for (int e=0; e<N_EXPERTS; e++) se += expf(sc[e]-mx);
                    float inv = 1.0f/(se+1e-30f);
                    float sm[256]; for(int e=0;e<N_EXPERTS;e++) sm[e]=expf(sc[e]-mx)*inv;
                    int *is = topk_idx + s*N_ACTIVE_EXPTS;
                    float *ws = topk_wt + s*N_ACTIVE_EXPTS;
                    for(int k=0;k<N_ACTIVE_EXPTS;k++){
                        int bi=-1; float bv=-1e30f;
                        for(int e=0;e<N_EXPERTS;e++){
                            int used=0; for(int pk=0;pk<k;pk++) if(is[pk]==e){used=1;break;}
                            if(!used&&sm[e]>bv){bv=sm[e];bi=e;}
                        }
                        is[k]=bi; ws[k]=bv;
                    }
                    float sw=0; for(int k=0;k<N_ACTIVE_EXPTS;k++) sw+=ws[k];
                    if(sw>1e-30f){float iv=1.0f/sw;for(int k=0;k<N_ACTIVE_EXPTS;k++)ws[k]*=iv;}
                    for(int k=0;k<N_ACTIVE_EXPTS;k++){
                        if(is[k]<0)continue;
                        int seen=0; for(int u=0;u<mc->n_unique;u++) if(mc->unique_ids[u]==is[k]){seen=1;break;}
                        if(!seen) mc->unique_ids[mc->n_unique++]=is[k];
                    }
                }
                
                // 4. Dequantize shared expert (once, cached)
                if (!mc->gate_shexp && mq->q_gate_shexp) {
                    mc->gate_shexp = (float *)malloc(D_MODEL*SHARED_D_FF*sizeof(float));
                    mc->up_shexp = (float *)malloc(D_MODEL*SHARED_D_FF*sizeof(float));
                    mc->down_shexp = (float *)malloc(SHARED_D_FF*D_MODEL*sizeof(float));
                    gguf_dequantize(mq->q_gate_shexp, mq->ty_gs, (int64_t)D_MODEL*SHARED_D_FF, mc->gate_shexp);
                    gguf_dequantize(mq->q_up_shexp, mq->ty_gs, (int64_t)D_MODEL*SHARED_D_FF, mc->up_shexp);
                    gguf_dequantize(mq->q_down_shexp, mq->ty_gs, (int64_t)SHARED_D_FF*D_MODEL, mc->down_shexp);
                }
                
                // 5. Dequantize unique experts + build moe_weights_t
                moe_weights_t mw; memset(&mw,0,sizeof(mw));
                mw.ffn_gate_inp = mc->deq_gate_inp;
                mw.ffn_gate_shexp = mc->gate_shexp; mw.ffn_up_shexp = mc->up_shexp; mw.ffn_down_shexp = mc->down_shexp;
                mw.loaded = true;
                
                float *ge = (float *)calloc((int64_t)N_EXPERTS*D_MODEL*D_FF,sizeof(float));
                float *ue = (float *)calloc((int64_t)N_EXPERTS*D_MODEL*D_FF,sizeof(float));
                float *de = (float *)calloc((int64_t)N_EXPERTS*D_FF*D_MODEL,sizeof(float));
                
                for (int u = 0; u < mc->n_unique; u++) {
                    int eid = mc->unique_ids[u];
                    mc->expert_gate[u] = (float *)malloc((int64_t)D_MODEL*D_FF*sizeof(float));
                    mc->expert_up[u] = (float *)malloc((int64_t)D_MODEL*D_FF*sizeof(float));
                    mc->expert_down[u] = (float *)malloc((int64_t)D_FF*D_MODEL*sizeof(float));
                    gguf_dequantize(mq->q_gate_exps+(int64_t)eid*mq->expert_raw, mq->ty_ge, (int64_t)D_MODEL*D_FF, mc->expert_gate[u]);
                    gguf_dequantize(mq->q_up_exps+(int64_t)eid*mq->expert_raw, mq->ty_ge, (int64_t)D_MODEL*D_FF, mc->expert_up[u]);
                    gguf_dequantize(mq->q_down_exps+(int64_t)eid*mq->expert_raw_down, mq->ty_ge, (int64_t)D_FF*D_MODEL, mc->expert_down[u]);
                    memcpy(ge+(int64_t)eid*D_MODEL*D_FF, mc->expert_gate[u], (int64_t)D_MODEL*D_FF*sizeof(float));
                    memcpy(ue+(int64_t)eid*D_MODEL*D_FF, mc->expert_up[u], (int64_t)D_MODEL*D_FF*sizeof(float));
                    memcpy(de+(int64_t)eid*D_FF*D_MODEL, mc->expert_down[u], (int64_t)D_FF*D_MODEL*sizeof(float));
                }
                mw.ffn_gate_exps = ge; mw.ffn_up_exps = ue; mw.ffn_down_exps = de;
                
                // 6. Run MoE forward
                wubu_moe_forward(n2, B, T, &mw, ffn_out);
                free(ge); free(ue); free(de); free(scores);
                
                // 7. Upload MoE output to GPU for residual add
                cudaMemcpyAsync(d_norm_p, ffn_out, N*D_MODEL*sizeof(float),
                               cudaMemcpyHostToDevice, stream);
            } else {
                // Identity pass-through (MoE disabled or no tensor)
                memcpy(saved_ffn_out + l * N * D_MODEL, saved_normed2 + l * N * D_MODEL,
                       N * D_MODEL * sizeof(float));
            }
            
            cublasSaxpy(cublas_h, N * D_MODEL, &alpha, d_norm_p, 1, d_cur, 1);
            cudaStreamSynchronize(stream);
        }
        
        // Final RMSNorm
        cudaMemcpyAsync(d_norm_weight, model.norm_weight,
                      D_MODEL * sizeof(float), cudaMemcpyHostToDevice, stream);
        wubu_cuda_rms_norm(B, T, D_MODEL, d_cur, d_norm_weight, 1e-6f, d_norm_p, stream);
        cudaMemcpy(d_cur, d_norm_p, N * D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice);
        
        cudaMemcpy(hidden, d_cur, N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(stream);
        
        // === CPU: Output projection + Loss + Gradient ===
        for (int i = 0; i < N; i++) {
            const float *h = hidden + i * D_MODEL;
            float *log_i = logits + i * vocab_size;
            for (int j = 0; j < vocab_size; j++) {
                double sum = 0.0;
                for (int k = 0; k < D_MODEL; k++)
                    sum += (double)h[k] * (double)output_weight[j * D_MODEL + k];
                log_i[j] = (float)sum;
            }
        }
        
        // CE loss + gradient w.r.t. logits
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
        for (int j = 0; j < vocab_size; j++)
            for (int k = 0; k < D_MODEL; k++) {
                double sum = 0.0;
                for (int i = 0; i < N; i++)
                    sum += (double)hidden[i * D_MODEL + k] * (double)dlogits[i * vocab_size + j];
                dW[j * D_MODEL + k] = (float)(sum / N);
            }
        
        // === Direct CPU Backward (deferred weight update) ===
        float *d_hidden = (float *)malloc(N * D_MODEL * sizeof(float));
        memset(d_hidden, 0, N * D_MODEL * sizeof(float));
        for (int i = 0; i < N; i++)
            for (int k = 0; k < D_MODEL; k++) {
                double sum = 0.0;
                for (int j = 0; j < vocab_size; j++)
                    sum += (double)dlogits[i * vocab_size + j] * (double)output_weight[j * D_MODEL + k];
                d_hidden[i * D_MODEL + k] = (float)sum;
            }
        
        double d_hidden_max_norm = 0.0;
        for (int s = 0; s < N; s++) {
            float *d_s = d_hidden + s * D_MODEL;
            double sq_sum = 0.0;
            for (int k = 0; k < D_MODEL; k++) sq_sum += (double)d_s[k] * (double)d_s[k];
            float norm = sqrtf((float)sq_sum);
            if ((double)norm > d_hidden_max_norm) d_hidden_max_norm = (double)norm;
            if (norm > 100.0f) {
                float scale = 100.0f / norm;
                for (int k = 0; k < D_MODEL; k++) d_s[k] *= scale;
            }
        }
        
        // Deferred weight gradient storage
        // Store [layer][wi] = {weight_ptr, grad_ptr, size}
        weight_grad_t deferred_w[40][16];
        weight_grad_t deferred_g[40][16];
        long long deferred_sz[40][16];
        int deferred_n[40] = {0};
        
        float *d_x_bwd = d_hidden;
        float *d_embd = (float *)calloc(N * D_MODEL, sizeof(float));
        
        for (int l = model.n_layers - 1; l >= 0; l--) {
            const wubu_layer_t *layer = &model.layers[l];
            
            // Allocate per-layer weight grad buffers
            float *d_qkv_weight = NULL, *d_gate_weight = NULL, *d_beta_weight = NULL;
            float *d_alpha_weight = NULL, *d_conv1d_weight = NULL, *d_ssm_out_weight = NULL;
            float *d_ssm_norm_weight = NULL, *d_ssm_state_init_grad = NULL;
            float *d_q_weight = NULL, *d_k_weight = NULL, *d_v_weight = NULL;
            float *d_q_norm_weight = NULL, *d_k_norm_weight = NULL, *d_out_weight = NULL;
            
            if (layer->is_ssm) {
                d_qkv_weight = (float *)calloc(D_MODEL * qkv_dim, sizeof(float));
                d_gate_weight = (float *)calloc(D_MODEL * VALUE_DIM, sizeof(float));
                d_beta_weight = (float *)calloc(D_MODEL * DT_RANK, sizeof(float));
                d_alpha_weight = (float *)calloc(D_MODEL * DT_RANK, sizeof(float));
                d_conv1d_weight = (float *)calloc(CONV_KERNEL * CONV_DIM, sizeof(float));
                d_ssm_out_weight = (float *)calloc(VALUE_DIM * D_MODEL, sizeof(float));
                d_ssm_norm_weight = (float *)calloc(SSM_D_STATE, sizeof(float));
                d_ssm_state_init_grad = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
            } else {
                d_q_weight = (float *)calloc(D_MODEL * q_dim_x2, sizeof(float));
                d_k_weight = (float *)calloc(D_MODEL * kv_dim, sizeof(float));
                d_v_weight = (float *)calloc(D_MODEL * kv_dim, sizeof(float));
                d_q_norm_weight = (float *)calloc(GQA_HEAD_DIM, sizeof(float));
                d_k_norm_weight = (float *)calloc(GQA_HEAD_DIM, sizeof(float));
                d_out_weight = (float *)calloc(gqa_q_dim * D_MODEL, sizeof(float));
            }
            
            float *d_normed = (float *)calloc(N * D_MODEL, sizeof(float));
            float *d_x_post = (float *)calloc(N * D_MODEL, sizeof(float));
            
            const float *normed2_inp = saved_normed2 + l * N * D_MODEL;
            
            // === Lazy MoE backward (uses forward-cached weights) ===
            if (moe_enabled && moe_cache[l].n_unique > 0) {
                lazy_moe_cache_t *mc = &moe_cache[l];
                const float *normed2_inp = saved_normed2 + l * N * D_MODEL;
                
                // Rebuild moe_weights_t from cached dequantized experts
                moe_weights_t mw; memset(&mw,0,sizeof(mw));
                mw.ffn_gate_inp = mc->deq_gate_inp;
                mw.ffn_gate_shexp = mc->gate_shexp;
                mw.ffn_up_shexp = mc->up_shexp;
                mw.ffn_down_shexp = mc->down_shexp;
                mw.loaded = true;
                
                float *ge = (float *)calloc((int64_t)N_EXPERTS*D_MODEL*D_FF,sizeof(float));
                float *ue = (float *)calloc((int64_t)N_EXPERTS*D_MODEL*D_FF,sizeof(float));
                float *de = (float *)calloc((int64_t)N_EXPERTS*D_FF*D_MODEL,sizeof(float));
                for (int u = 0; u < mc->n_unique; u++) {
                    int eid = mc->unique_ids[u];
                    memcpy(ge+(int64_t)eid*D_MODEL*D_FF, mc->expert_gate[u], (int64_t)D_MODEL*D_FF*sizeof(float));
                    memcpy(ue+(int64_t)eid*D_MODEL*D_FF, mc->expert_up[u], (int64_t)D_MODEL*D_FF*sizeof(float));
                    memcpy(de+(int64_t)eid*D_FF*D_MODEL, mc->expert_down[u], (int64_t)D_FF*D_MODEL*sizeof(float));
                }
                mw.ffn_gate_exps = ge; mw.ffn_up_exps = ue; mw.ffn_down_exps = de;
                
                // Allocate MoE weight gradients
                float *d_normed2 = (float *)malloc(N * D_MODEL * sizeof(float));
                float *d_gate_inp = (float *)calloc(D_MODEL*N_EXPERTS, sizeof(float));
                float *d_gate_exps = (float *)calloc((int64_t)N_EXPERTS*D_MODEL*D_FF, sizeof(float));
                float *d_up_exps = (float *)calloc((int64_t)N_EXPERTS*D_MODEL*D_FF, sizeof(float));
                float *d_down_exps = (float *)calloc((int64_t)N_EXPERTS*D_FF*D_MODEL, sizeof(float));
                float *d_gs = (float *)calloc((int64_t)D_MODEL*SHARED_D_FF, sizeof(float));
                float *d_us = (float *)calloc((int64_t)D_MODEL*SHARED_D_FF, sizeof(float));
                float *d_ds = (float *)calloc((int64_t)SHARED_D_FF*D_MODEL, sizeof(float));
                
                wubu_moe_backward(d_x_bwd, B, T, normed2_inp, &mw,
                    d_normed2, d_gate_inp, d_gate_exps, d_up_exps, d_down_exps,
                    d_gs, d_us, d_ds);
                
                // Post-attention RMSNorm backward
                wubu_rms_norm_backward(B, T, D_MODEL, normed2_inp, layer->post_attn_norm_weight,
                                       1e-6f, d_normed2, d_x_post);
                for (int i = 0; i < N * D_MODEL; i++) d_x_post[i] += d_x_bwd[i];
                
                // Free + dealloc temp arrays
                free(ge); free(ue); free(de);
                free(d_gate_inp); free(d_gate_exps); free(d_up_exps); free(d_down_exps);
                free(d_gs); free(d_us); free(d_ds);
                
                // Free cached MoE expert weights (no longer needed)
                for (int u = 0; u < mc->n_unique; u++) {
                    free(mc->expert_gate[u]); free(mc->expert_up[u]); free(mc->expert_down[u]);
                    mc->expert_gate[u] = NULL; mc->expert_up[u] = NULL; mc->expert_down[u] = NULL;
                }
                mc->n_unique = 0;
                // Keep deq_gate_inp and shared expert for reuse (freed at end)
            } else {
                // Identity MoE backward (unchanged)
                float *d_normed2 = (float *)malloc(N * D_MODEL * sizeof(float));
                memcpy(d_normed2, d_x_bwd, N * D_MODEL * sizeof(float));
                wubu_rms_norm_backward(B, T, D_MODEL, normed2_inp, layer->post_attn_norm_weight,
                                       1e-6f, d_normed2, d_x_post);
                for (int i = 0; i < N * D_MODEL; i++) d_x_post[i] += d_x_bwd[i];
                free(d_normed2);
            }
            
            float *d_attn_out = d_x_post;
            
            // Backward call with non-NULL weight grads
            if (layer->is_ssm && cpu_ssm_scr && cpu_ssm_scr[l]) {
                float *p = cpu_ssm_scr[l];
                float *cpu_qkv    = p; p += N * qkv_dim;
                float *cpu_z      = p; p += N * VALUE_DIM;
                float *cpu_beta_r = p; p += N * DT_RANK;
                float *cpu_alpha_r= p; p += N * DT_RANK;
                float *cpu_beta_s = p; p += N * DT_RANK;
                float *cpu_gate   = p; p += N * DT_RANK;
                float *cpu_conv   = p; p += N * CONV_DIM;
                float *cpu_q_c    = p; p += N * KEY_DIM;
                float *cpu_k_c    = p; p += N * KEY_DIM;
                float *cpu_v_c    = p; p += N * VALUE_DIM;
                float *cpu_q_n    = p; p += N * KEY_DIM;
                float *cpu_k_n    = p; p += N * KEY_DIM;
                float *cpu_delta  = p; p += N * VALUE_DIM;
                float *cpu_z_s    = p; p += N * VALUE_DIM;
                float *cpu_states = p; p += (T+1) * state_sz;
                float *cpu_conv_s = p;
                
                const float *normed_inp = saved_normed + l * N * D_MODEL;
                const float *attn_out_saved = saved_attn_out + l * N * D_MODEL;
                
                wubu_ssm_backward(B, T, normed_inp, attn_out_saved, d_attn_out,
                                  cpu_qkv, cpu_z, cpu_beta_r, cpu_alpha_r,
                                  cpu_conv, cpu_q_c, cpu_k_c, cpu_v_c,
                                  cpu_q_n, cpu_k_n, cpu_delta, cpu_z_s,
                                  cpu_states, cpu_beta_s, cpu_gate,
                                  cpu_conv_s, &layer->ssm,
                                  d_normed,
                                  d_qkv_weight, d_gate_weight,
                                  d_beta_weight, d_alpha_weight,
                                  d_conv1d_weight, d_ssm_out_weight,
                                  d_ssm_norm_weight, d_ssm_state_init_grad);
                
                for (int s = 0; s < N; s++) {
                    float *d_s = d_normed + s * D_MODEL;
                    double sq_sum = 0.0;
                    for (int k = 0; k < D_MODEL; k++) sq_sum += (double)d_s[k] * (double)d_s[k];
                    float norm = sqrtf((float)sq_sum);
                    if (norm > 10.0f) {
                        float scale = 10.0f / norm;
                        for (int k = 0; k < D_MODEL; k++) d_s[k] *= scale;
                    }
                }
                
                // Store SSM grads for deferred update (append after MoE)
                int wi = deferred_n[l];
                deferred_w[l][wi].ptr = layer->ssm.attn_qkv_weight; deferred_g[l][wi].ptr = d_qkv_weight; deferred_sz[l][wi] = D_MODEL * qkv_dim; wi++;
                deferred_w[l][wi].ptr = layer->ssm.attn_gate_weight; deferred_g[l][wi].ptr = d_gate_weight; deferred_sz[l][wi] = D_MODEL * VALUE_DIM; wi++;
                deferred_w[l][wi].ptr = layer->ssm.ssm_beta_weight; deferred_g[l][wi].ptr = d_beta_weight; deferred_sz[l][wi] = D_MODEL * DT_RANK; wi++;
                deferred_w[l][wi].ptr = layer->ssm.ssm_alpha_weight; deferred_g[l][wi].ptr = d_alpha_weight; deferred_sz[l][wi] = D_MODEL * DT_RANK; wi++;
                deferred_w[l][wi].ptr = layer->ssm.ssm_conv1d_weight; deferred_g[l][wi].ptr = d_conv1d_weight; deferred_sz[l][wi] = CONV_KERNEL * CONV_DIM; wi++;
                deferred_w[l][wi].ptr = layer->ssm.ssm_out_weight; deferred_g[l][wi].ptr = d_ssm_out_weight; deferred_sz[l][wi] = VALUE_DIM * D_MODEL; wi++;
                deferred_w[l][wi].ptr = layer->ssm.ssm_norm_weight; deferred_g[l][wi].ptr = d_ssm_norm_weight; deferred_sz[l][wi] = SSM_D_STATE; wi++;
                deferred_n[l] = wi;
            } else if (!layer->is_ssm && cpu_gqa_scr && cpu_gqa_scr[l]) {
                float *p = cpu_gqa_scr[l];
                float *cpu_Q_full  = p; p += N * q_dim_x2;
                float *cpu_K_raw   = p; p += N * kv_dim;
                float *cpu_Q_norm  = p; p += N * gqa_q_dim;
                float *cpu_K_norm  = p; p += N * kv_dim;
                float *cpu_V       = p; p += N * kv_dim;
                float *cpu_attn_out = p; p += N * gqa_q_dim;
                
                float *gate_sig = (float *)malloc(N * gqa_q_dim * sizeof(float));
                for (int i = 0; i < N * gqa_q_dim; i++)
                    gate_sig[i] = 1.0f / (1.0f + expf(-cpu_Q_full[gqa_q_dim + i]));
                
                const float *normed_inp = saved_normed + l * N * D_MODEL;
                const float *attn_out_saved = saved_attn_out + l * N * D_MODEL;
                
                wubu_gqa_backward(B, T, normed_inp,
                    cpu_Q_norm, cpu_Q_full, cpu_K_norm, cpu_K_raw,
                    cpu_V,
                    cpu_Q_full + gqa_q_dim, gate_sig,
                    cpu_attn_out, attn_out_saved,
                    d_attn_out, &layer->gqa, d_normed,
                    d_q_weight, d_k_weight, d_v_weight,
                    d_q_norm_weight, d_k_norm_weight, d_out_weight);
                
                free(gate_sig);
                
                for (int s = 0; s < N; s++) {
                    float *d_s = d_normed + s * D_MODEL;
                    double sq_sum = 0.0;
                    for (int k = 0; k < D_MODEL; k++) sq_sum += (double)d_s[k] * (double)d_s[k];
                    float norm = sqrtf((float)sq_sum);
                    if (norm > 10.0f) {
                        float scale = 10.0f / norm;
                        for (int k = 0; k < D_MODEL; k++) d_s[k] *= scale;
                    }
                }
                
                // Store GQA grads for deferred update (append after MoE)
                int wi = deferred_n[l];
                deferred_w[l][wi].ptr = layer->gqa.attn_q_weight; deferred_g[l][wi].ptr = d_q_weight; deferred_sz[l][wi] = D_MODEL * q_dim_x2; wi++;
                deferred_w[l][wi].ptr = layer->gqa.attn_k_weight; deferred_g[l][wi].ptr = d_k_weight; deferred_sz[l][wi] = D_MODEL * kv_dim; wi++;
                deferred_w[l][wi].ptr = layer->gqa.attn_v_weight; deferred_g[l][wi].ptr = d_v_weight; deferred_sz[l][wi] = D_MODEL * kv_dim; wi++;
                deferred_w[l][wi].ptr = layer->gqa.attn_q_norm_weight; deferred_g[l][wi].ptr = d_q_norm_weight; deferred_sz[l][wi] = GQA_HEAD_DIM; wi++;
                deferred_w[l][wi].ptr = layer->gqa.attn_k_norm_weight; deferred_g[l][wi].ptr = d_k_norm_weight; deferred_sz[l][wi] = GQA_HEAD_DIM; wi++;
                deferred_w[l][wi].ptr = layer->gqa.attn_output_weight; deferred_g[l][wi].ptr = d_out_weight; deferred_sz[l][wi] = gqa_q_dim * D_MODEL; wi++;
                deferred_n[l] = wi;
            } else {
                memcpy(d_normed, d_attn_out, N * D_MODEL * sizeof(float));
            }
            
            // Pre-attention RMSNorm backward
            float *d_x_pre = (float *)calloc(N * D_MODEL, sizeof(float));
            const float *normed_inp = saved_normed + l * N * D_MODEL;
            wubu_rms_norm_backward(B, T, D_MODEL, normed_inp, layer->attn_norm_weight,
                                   1e-6f, d_normed, d_x_pre);
            for (int i = 0; i < N * D_MODEL; i++) d_x_pre[i] += d_x_post[i];
            
            if (l > 0) memcpy(d_x_bwd, d_x_pre, N * D_MODEL * sizeof(float));
            else memcpy(d_embd, d_x_pre, N * D_MODEL * sizeof(float));
            
            free(d_normed); free(d_x_post); free(d_x_pre);
        }
        free(d_hidden);
        free(d_embd);
        
        // === Batch Weight Update (all layers, OpenMP) ===
        float max_g = 0.0f;
        // Find global max gradient
        #pragma omp parallel for reduction(max:max_g)
        for (int l = 0; l < model.n_layers; l++) {
            for (int wi = 0; wi < deferred_n[l]; wi++) {
                long long sz = deferred_sz[l][wi];
                float *g = deferred_g[l][wi].ptr;
                for (long long j = 0; j < sz; j++) {
                    float abs_g = fabsf(g[j]);
                    if (abs_g > max_g) max_g = abs_g;
                }
            }
        }
        
        // Pass 2: apply SGD with per-element gradient clipping at 10.0
        #pragma omp parallel for
        for (int l = 0; l < model.n_layers; l++) {
            for (int wi = 0; wi < deferred_n[l]; wi++) {
                long long sz = deferred_sz[l][wi];
                float *w = deferred_w[l][wi].ptr;
                float *g = deferred_g[l][wi].ptr;
                for (long long j = 0; j < sz; j++) {
                    float gv = g[j];
                    // TGT: wrap gradient through π-odometer instead of clipping
                    // remainder = fmod(g + pi, 2pi) - pi preserves direction
                    // While clipping [-10,10] loses directional info, TGT preserves it
                    float remainder = fmodf(gv + (float)M_PI, 2.0f * (float)M_PI) - (float)M_PI;
                    // Use remainder (stable direction) scaled by learning rate
                    // The quotient (magnitude wraps) is implicitly accumulated in weights
                    w[j] -= lr * remainder;
                }
            }
        }
        
        // Sync all updated weights to GPU (sequential stream)
        for (int l = 0; l < model.n_layers; l++) {
            if (model.layers[l].is_ssm) {
                float *cpu_w = model.layers[l].ssm.attn_qkv_weight;
                cudaMemcpyAsync(ssm_gpu_weights[l].d_attn_qkv, cpu_w,
                    D_MODEL * qkv_dim * sizeof(float), cudaMemcpyHostToDevice, stream);
                cpu_w = model.layers[l].ssm.attn_gate_weight;
                cudaMemcpyAsync(ssm_gpu_weights[l].d_attn_gate, cpu_w,
                    D_MODEL * VALUE_DIM * sizeof(float), cudaMemcpyHostToDevice, stream);
                cpu_w = model.layers[l].ssm.ssm_beta_weight;
                cudaMemcpyAsync(ssm_gpu_weights[l].d_ssm_beta, cpu_w,
                    D_MODEL * DT_RANK * sizeof(float), cudaMemcpyHostToDevice, stream);
                cpu_w = model.layers[l].ssm.ssm_alpha_weight;
                cudaMemcpyAsync(ssm_gpu_weights[l].d_ssm_alpha, cpu_w,
                    D_MODEL * DT_RANK * sizeof(float), cudaMemcpyHostToDevice, stream);
                cpu_w = model.layers[l].ssm.ssm_conv1d_weight;
                cudaMemcpyAsync(ssm_gpu_weights[l].d_ssm_conv1d, cpu_w,
                    CONV_KERNEL * CONV_DIM * sizeof(float), cudaMemcpyHostToDevice, stream);
                cpu_w = model.layers[l].ssm.ssm_out_weight;
                cudaMemcpyAsync(ssm_gpu_weights[l].d_ssm_out, cpu_w,
                    VALUE_DIM * D_MODEL * sizeof(float), cudaMemcpyHostToDevice, stream);
                cpu_w = model.layers[l].ssm.ssm_norm_weight;
                cudaMemcpyAsync(ssm_gpu_weights[l].d_ssm_norm, cpu_w,
                    SSM_D_STATE * sizeof(float), cudaMemcpyHostToDevice, stream);
            } else {
                float *cpu_w = model.layers[l].gqa.attn_q_weight;
                cudaMemcpyAsync(gqa_gpu_weights[l].d_attn_q, cpu_w,
                    D_MODEL * q_dim_x2 * sizeof(float), cudaMemcpyHostToDevice, stream);
                cpu_w = model.layers[l].gqa.attn_k_weight;
                cudaMemcpyAsync(gqa_gpu_weights[l].d_attn_k, cpu_w,
                    D_MODEL * kv_dim * sizeof(float), cudaMemcpyHostToDevice, stream);
                cpu_w = model.layers[l].gqa.attn_v_weight;
                cudaMemcpyAsync(gqa_gpu_weights[l].d_attn_v, cpu_w,
                    D_MODEL * kv_dim * sizeof(float), cudaMemcpyHostToDevice, stream);
                cpu_w = model.layers[l].gqa.attn_q_norm_weight;
                cudaMemcpyAsync(gqa_gpu_weights[l].d_q_norm_w, cpu_w,
                    GQA_HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice, stream);
                cpu_w = model.layers[l].gqa.attn_k_norm_weight;
                cudaMemcpyAsync(gqa_gpu_weights[l].d_k_norm_w, cpu_w,
                    GQA_HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice, stream);
                cpu_w = model.layers[l].gqa.attn_output_weight;
                cudaMemcpyAsync(gqa_gpu_weights[l].d_attn_out_w, cpu_w,
                    gqa_q_dim * D_MODEL * sizeof(float), cudaMemcpyHostToDevice, stream);
            }
        }
        
        // Free all deferred grad buffers
        for (int l = 0; l < model.n_layers; l++) {
            if (deferred_n[l] > 0) {
                // Only free if layer has SSM-like grads (check via is_ssm)
                for (int wi = 0; wi < deferred_n[l]; wi++)
                    free(deferred_g[l][wi].ptr);
            }
        }
        
        // Output weight SGD
        float max_g_out = 0.0f;
        for (int64_t idx = 0; idx < (int64_t)D_MODEL * vocab_size; idx++) {
            float v = fabsf(dW[idx]);
            if (v > max_g_out) max_g_out = v;
        }
        float clip_out = max_g_out > 1.0f ? 1.0f / max_g_out : 1.0f;
        for (int64_t idx = 0; idx < (int64_t)D_MODEL * vocab_size; idx++)
            output_weight[idx] -= lr * dW[idx] * clip_out;
        
        // Q-learner: update LR for next step
        float new_lr = qlearner_step(&ql, loss);
        
        cudaStreamSynchronize(stream);
        double step_time = now_sec() - t0;
        total_time += step_time;
        
        printf("Step %3d: loss=%.4f (%.3fs, %.1f tok/s) | dH_max=%.1e gW_max=%.1e qlr=%.6f\n",
               step + 1, loss, step_time, N / step_time,
               d_hidden_max_norm, max_g, new_lr);
        fflush(stdout);
    }
    gguf_close(gguf_moe);
    
    printf("\n=== RESULTS ===\n");
    printf("Avg time/step: %.3fs (%.1f tok/s)\n",
           total_time / n_steps, N / (total_time / n_steps));
    printf("Output weight + all %d internal layers trained via SGD (Q-learner adaptive LR)\n",
           model.n_layers);
    
    // Cleanup (same as original)
    for (int l = 0; l < model.n_layers; l++) {
        if (model.layers[l].is_ssm) gpu_free_ssm_weights(&ssm_gpu_weights[l]);
        else gpu_free_gqa_weights(&gqa_gpu_weights[l]);
    }
    free(ssm_gpu_weights); free(gqa_gpu_weights);
    for (int l = 0; l < model.n_layers; l++) {
        wubu_cuda_free(d_ssm_states[l]); wubu_cuda_free(d_conv_states[l]);
        if (d_states_t && d_states_t[l]) wubu_cuda_free(d_states_t[l]);
        if (cpu_ssm_scr && cpu_ssm_scr[l]) free(cpu_ssm_scr[l]);
    }
    free(d_ssm_states); free(d_conv_states); free(d_states_t); free(cpu_ssm_scr);
    for (int l = 0; l < model.n_layers; l++) {
        if (!model.layers[l].is_ssm) {
            if (d_gqa_q_norm_save[l]) wubu_cuda_free(d_gqa_q_norm_save[l]);
            if (d_gqa_k_raw_save[l]) wubu_cuda_free(d_gqa_k_raw_save[l]);
            if (cpu_gqa_scr[l]) free(cpu_gqa_scr[l]);
        }
    }
    free(d_gqa_q_norm_save); free(d_gqa_k_raw_save); free(cpu_gqa_scr);
    wubu_cuda_free(d_x); wubu_cuda_free(d_out); wubu_cuda_free(d_norm);
    wubu_cuda_free(d_norm_weight); wubu_cuda_free(d_poincare_norms);
    // Free SSM scratch
    wubu_cuda_free(ssm_scr.d_qkv); wubu_cuda_free(ssm_scr.d_z);
    wubu_cuda_free(ssm_scr.d_beta); wubu_cuda_free(ssm_scr.d_alpha);
    wubu_cuda_free(ssm_scr.d_beta_sig); wubu_cuda_free(ssm_scr.d_alpha_bi);
    wubu_cuda_free(ssm_scr.d_gate); wubu_cuda_free(ssm_scr.d_conv_input);
    wubu_cuda_free(ssm_scr.d_conv_out); wubu_cuda_free(ssm_scr.d_q_conv);
    wubu_cuda_free(ssm_scr.d_k_conv); wubu_cuda_free(ssm_scr.d_v_conv);
    wubu_cuda_free(ssm_scr.d_q_norm); wubu_cuda_free(ssm_scr.d_k_norm);
    wubu_cuda_free(ssm_scr.d_delta_out); wubu_cuda_free(ssm_scr.d_z_silu);
    // Free GQA scratch
    wubu_cuda_free(gqa_scr.d_Q_full); wubu_cuda_free(gqa_scr.d_K);
    wubu_cuda_free(gqa_scr.d_V); wubu_cuda_free(gqa_scr.d_scratch);
    wubu_cuda_destroy(cublas_h, stream);
    free(tokens); free(embd); free(hidden); free(logits);
    free(dlogits); free(dW); free(output_weight); free(norm_weight_buf);
    free(hidden_per_layer); free(saved_normed); free(saved_attn_out);
    free(saved_normed2); free(saved_ffn_out);
    wubu_tokenizer_free(&tok);
    wubu_model_free(&model);
    return 0;
}
