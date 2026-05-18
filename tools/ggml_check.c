/* Verify SSM forward by calling ggml operators directly (same as llama.cpp)
 * Link against: -lggml-base -lggml-cpu -lggml */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#define D_MODEL 2048
#define SSM_K_HEADS 16
#define SSM_V_HEADS 32
#define SSM_D_STATE 128
#define KEY_DIM (SSM_D_STATE * SSM_K_HEADS)
#define VALUE_DIM (SSM_D_STATE * SSM_V_HEADS)
#define CONV_DIM (KEY_DIM * 2 + VALUE_DIM)
#define DT_RANK 32

// Write a float array to file
void save_bin(const char *path, const float *data, size_t n) {
    FILE *f = fopen(path, "wb");
    fwrite(data, sizeof(float), n, f);
    fclose(f);
}

int main() {
    // Load model weights from binary files dumped by dump_weights_l0
    // We need: 
    // - attn_qkv_weight [D_MODEL, CONV_DIM]
    // - attn_gate_weight [D_MODEL, VALUE_DIM]
    // - ssm_beta_weight [D_MODEL, DT_RANK]
    // - ssm_alpha_weight [D_MODEL, DT_RANK]
    // - ssm_conv1d_weight [CONV_KERNEL, CONV_DIM]
    // - ssm_norm_weight [SSM_D_STATE]
    // - ssm_out_weight [VALUE_DIM, D_MODEL]
    // - ssm_dt_bias [DT_RANK]
    // - ssm_a [DT_RANK]
    
    // Embedding
    float *emb = (float*)malloc(D_MODEL * sizeof(float));
    FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
    if (!f) { fprintf(stderr, "No emb file\n"); return 1; }
    fseek(f, 248044LL * D_MODEL * sizeof(float), SEEK_SET);
    fread(emb, sizeof(float), D_MODEL, f);
    fclose(f);
    
    printf("Emb: mean=%.6f std=%.6f\n", 
           mean(emb, D_MODEL), stddev(emb, D_MODEL));
    
    // Now let's use our C code to do the forward and dump EVERY intermediate
    // We already have this from dump_intermediates
    
    // Actually let's use the ggml library directly.
    // Initialize ggml
    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024 * 1024,  // 1GB
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }
    
    // Create tensors for everything
    struct ggml_tensor *emb_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D_MODEL);
    memcpy(emb_t->data, emb, D_MODEL * sizeof(float));
    
    printf("ggml_init OK\n");
    
    ggml_free(ctx);
    free(emb);
    return 0;
}

static double mean(const float *x, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += x[i]; return s / n;
}
static double stddev(const float *x, int n) {
    double m = mean(x, n), s = 0;
    for (int i = 0; i < n; i++) s += (x[i] - m) * (x[i] - m);
    return sqrt(s / n);
}
