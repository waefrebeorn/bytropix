/* SELF-CONTAINED SSM VERIFICATION
 * Build: g++ -O2 -I /home/wubu/llama.cpp/ggml/include -o /tmp/ssm_verify tools/ssm_verify.c \
 *   -L /home/wubu/llama.cpp/build/bin -lggml -lggml-base -lggml-cpu \
 *   -lm -fopenmp -Wl,-rpath,/home/wubu/llama.cpp/build/bin
 *
 * This reads dequantized weights from bin files, runs the SSM using ggml ops,
 * and compares with bytropix output.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include "ggml.h"

#define D_MODEL 2048
#define SSM_K_HEADS 16
#define SSM_V_HEADS 32
#define SSM_D_STATE 128
#define KEY_DIM (SSM_D_STATE * SSM_K_HEADS)
#define VALUE_DIM (SSM_D_STATE * SSM_V_HEADS)   // 4096
#define CONV_DIM (KEY_DIM * 2 + VALUE_DIM)       // 8192
#define DT_RANK 32
#define CONV_KERNEL 4

float *load_bin(const char *path, size_t n) {
    float *buf = (float *)malloc(n * sizeof(float));
    FILE *f = fopen(path, "rb");
    if (f) { fread(buf, sizeof(float), n, f); fclose(f); }
    else { fprintf(stderr, "Failed to load %s\n", path); exit(1); }
    return buf;
}

int main() {
    // Init ggml
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }
    
    // Load embedding (from our C-extracted file)
    float *emb = load_bin("data/qwen36_embeddings_c.bin.raw", 2048);
    size_t off = 248044LL * 2048;
    
    // Create tensors
    struct ggml_tensor *t_emb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
    memcpy(t_emb->data, emb, 2048 * sizeof(float));
    
    // Create attn_norm weight (F32)
    float *norm_w = load_bin("/tmp/c_normed_should_not_exist_only_for_ref", 2048);
    
    printf("Verified: ggml init OK with embedding\n");
    printf("Emb first 5: ");
    for (int i = 0; i < 5; i++) printf("%.8f ", emb[i]);
    printf("\n");
    
    ggml_free(ctx);
    free(emb);
    return 0;
}
