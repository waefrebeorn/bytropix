/* Compare RMSNorm weights directly */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gguf_reader.h"
#include "llama.h"

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int D = 2048;
    
    // ---- Our reader ----
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    
    // Our attn_norm.weight for layer 0
    float *our_attn_norm = (float *)malloc(D * sizeof(float));
    float *our_post_attn_norm = (float *)malloc(D * sizeof(float));
    float *our_output_norm = (float *)malloc(D * sizeof(float));
    
    gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0.attn_norm.weight");
    if (t) gguf_read_tensor_f32(ctx, t, our_attn_norm, D);
    else printf("No attn_norm\n");
    
    t = gguf_find_tensor(ctx, "blk.0.post_attention_norm.weight");
    if (t) gguf_read_tensor_f32(ctx, t, our_post_attn_norm, D);
    else printf("No post_attn_norm\n");
    
    t = gguf_find_tensor(ctx, "output_norm.weight");
    if (t) gguf_read_tensor_f32(ctx, t, our_output_norm, D);
    else printf("No output_norm\n");
    
    gguf_close(ctx);
    
    // Print stats
    printf("Our blk.0.attn_norm: first=%.8f mean=%.8f\n", our_attn_norm[0], our_attn_norm[1]);
    printf("Our blk.0.post_attn_norm: first=%.8f second=%.8f\n", our_post_attn_norm[0], our_post_attn_norm[1]);
    printf("Our output_norm: first=%.8f second=%.8f\n", our_output_norm[0], our_output_norm[1]);
    
    // ---- llama.cpp ----
    // We need to access llama's internal model tensors
    // The cleanest way is to use gguf library that llama provides
    
    // For now, let's just read the same tensors from the GGUF using a second reader
    gguf_ctx *ctx2 = gguf_open(path);
    if (!ctx2) return 1;
    gguf_buffer_data(ctx2);
    
    float *ref_attn_norm = (float *)malloc(D * sizeof(float));
    t = gguf_find_tensor(ctx2, "blk.0.attn_norm.weight");
    if (t) gguf_read_tensor_f32(ctx2, t, ref_attn_norm, D);
    
    // Compare
    double max_diff = 0;
    int max_idx = -1;
    for (int i = 0; i < D; i++) {
        double diff = fabs(our_attn_norm[i] - ref_attn_norm[i]);
        if (diff > max_diff) { max_diff = diff; max_idx = i; }
    }
    printf("attn_norm vs self: max_diff=%.10f at idx=%d\n", max_diff, max_idx);
    
    // They should be identical since we read from the same file
    // But this just verifies our reader is deterministic
    
    // Now let's check what llama.cpp reads for these weights
    // by using the llama model API
    llama_backend_init();
    struct llama_model_params lmp = llama_model_default_params();
    lmp.n_gpu_layers = 0;
    struct llama_model *lm = llama_model_load_from_file(path, lmp);
    
    // We can't access raw tensors from llama API
    // But let's verify by comparing n_embd, n_layer etc
    int n_embd = llama_model_n_embd(lm);
    int n_layer = llama_model_n_layer(lm);
    int n_vocab = llama_model_n_vocab(lm);
    printf("llama: n_embd=%d n_layer=%d n_vocab=%d\n", n_embd, n_layer, n_vocab);
    
    llama_model_free(lm);
    llama_backend_free();
    
    free(our_attn_norm); free(our_post_attn_norm); free(our_output_norm);
    free(ref_attn_norm);
    return 0;
}
