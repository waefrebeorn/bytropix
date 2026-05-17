/**
 * compare_layer0.c — Run BOS token through 1 layer, dump output from both
 * our reader and llama.cpp. Compare cos-sim.
 * 
 * Build:
 * gcc -O2 -I include -o compare_layer0 tools/compare_layer0.c \
 *     src/gguf_reader.o src/wubu_ssm.o src/wubu_mobius.o src/wubu_moe.o \
 *     src/wubu_tokenizer.o src/qlearner.o -lm -fopenmp
 *
 * Usage: ./compare_layer0 model.gguf
 */
#include "gguf_reader.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s model.gguf\n", argv[0]); return 1; }
    
    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "Failed to open model\n"); return 1; }
    
    // Load BOS embedding
    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t) { fprintf(stderr, "No token_embd\n"); return 1; }
    
    // Find BOS token ID - try reading from model file
    // Default BOS for Qwen3: 248044 or check metadata later
    // For now, we read the embedding for token 0 (first token in vocab)
    float *embd = (float *)malloc(D_MODEL * sizeof(float));
    
    // Read token 0 (BOS-like) embedding: first D_MODEL values from token_embd
    // The tensor is [D_MODEL, vocab_size]. dims[0] innermost.
    // Token 0's embedding = first D_MODEL float values (rows of dims[0])
    // Actually with dims[0]=2048 innermost: embd[0] = tok0_dim0, embd[1] = tok0_dim1, ...
    // embd[2048] = tok1_dim0, embd[2049] = tok1_dim1, ...
    // But since we loaded the WHOLE tensor with gguf_read_tensor_f32, it's one big array.
    
    int64_t n_elems = 1;
    for (int d = 0; d < t->n_dims; d++) n_elems *= t->dims[d];
    float *all_embd = (float *)malloc(n_elems * sizeof(float));
    if (!gguf_read_tensor_f32(ctx, t, all_embd, n_elems)) {
        fprintf(stderr, "Failed to read token_embd\n"); return 1;
    }
    
    // Token 0's embedding is at offset 0 * D_MODEL = 0
    memcpy(embd, all_embd, D_MODEL * sizeof(float));
    printf("BOS embd[0:5]: %.6f %.6f %.6f %.6f %.6f\n", embd[0], embd[1], embd[2], embd[3], embd[4]);
    printf("BOS embd rms: %.6f\n", sqrtf(1.0f/D_MODEL * ({
        double s=0; for(int i=0;i<D_MODEL;i++) s+=embd[i]*embd[i]; s;
    })));
    
    // Load layer 0 SSM weights
    ssm_layer_weights sw;
    memset(&sw, 0, sizeof(sw));
    char name[256];
    
    snprintf(name, sizeof(name), "blk.0.attn_qkv.weight");
    t = gguf_find_tensor(ctx, name);
    if (t) {
        int64_t n = D_MODEL * CONV_DIM; // 2048 * 8192 = 16.7M
        sw.attn_qkv_weight = (float *)malloc(n * sizeof(float));
        gguf_read_tensor_f32(ctx, t, sw.attn_qkv_weight, n);
    }
    
    snprintf(name, sizeof(name), "blk.0.attn_gate.weight");
    t = gguf_find_tensor(ctx, name);
    if (t) {
        int64_t n = D_MODEL * VALUE_DIM;
        sw.attn_gate_weight = (float *)malloc(n * sizeof(float));
        gguf_read_tensor_f32(ctx, t, sw.attn_gate_weight, n);
    }
    
    // ... (load other SSM weights similarly)
    
    // For now, just do a simple test: dump normed vs expected
    float attn_norm_weight[D_MODEL];
    snprintf(name, sizeof(name), "blk.0.attn_norm.weight");
    t = gguf_find_tensor(ctx, name);
    if (t) gguf_read_tensor_f32(ctx, t, attn_norm_weight, D_MODEL);
    
    float normed[D_MODEL];
    wubu_rms_norm(1, 1, D_MODEL, embd, attn_norm_weight, 1e-6f, normed);
    printf("normed[0:5]: %.6f %.6f %.6f %.6f %.6f\n", normed[0], normed[1], normed[2], normed[3], normed[4]);
    printf("normed rms: %.6f\n", sqrtf(1.0f/D_MODEL * ({
        double s=0; for(int i=0;i<D_MODEL;i++) s+=normed[i]*normed[i]; s;
    })));
    
    // Now load attn_qkv_weight and compute QKV projection
    // QKV project: normed [2048] @ W_qkv [2048, 8192] -> qkv [8192]
    if (sw.attn_qkv_weight) {
        float qkv[CONV_DIM];
        #pragma omp parallel for
        for (int j = 0; j < CONV_DIM; j++) {
            double sum = 0.0;
            for (int i = 0; i < D_MODEL; i++)
                sum += normed[i] * sw.attn_qkv_weight[i + j * D_MODEL];
            qkv[j] = (float)sum;
        }
        printf("QKV[0:5]: %.6f %.6f %.6f %.6f %.6f\n", qkv[0], qkv[1], qkv[2], qkv[3], qkv[4]);
        printf("QKV rms: %.6f\n", sqrtf(1.0f/CONV_DIM * ({
            double s=0; for(int i=0;i<CONV_DIM;i++) s+=qkv[i]*qkv[i]; s;
        })));
    }
    
    free(all_embd);
    free(embd);
    free(sw.attn_qkv_weight);
    free(sw.attn_gate_weight);
    gguf_close(ctx);
    return 0;
}
