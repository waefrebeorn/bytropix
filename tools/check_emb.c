// Verify token embedding for BOS (248044) matches reference
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gguf_reader.h"

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const int D = 2048;
    const int BOS_ID = 248044;
    
    // Read token embeddings directly from GGUF
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    
    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t) { printf("ERROR: token_embd.weight not found\n"); return 1; }
    
    printf("token_embd.weight: dims=[%ld,%ld] type=%d\n",
           (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    printf("BOS offset: %ld floats\n", (long)BOS_ID * D);
    
    // Read just the BOS embedding
    float *bos = (float *)malloc((int64_t)D * sizeof(float));
    // Seek directly to BOS embedding in the tensor data
    // token_embd.weight has dims[0]=2048, so each token is 2048 floats
    // We need to read from offset BOS_ID * 2048 * sizeof(float) into the tensor data
    // 
    // Actually, let's read all token embeds from the GGUF tensor directly
    // The tensor has dims=[2048, 248320], storing layout is [vocab, dim] in memory
    // So total elems = 2048 * 248320 = 508559360
    // We can't buffer all of that. Read just the BOS token.
    
    // gguf_read_tensor_f32 reads from the beginning. We need a different approach:
    // Read the first BOS_ID+1 tokens = (BOS_ID+1)*D floats
    int64_t read_elems = (int64_t)(BOS_ID + 1) * D;
    float *all_bos = (float *)malloc(read_elems * sizeof(float));
    if (!all_bos) { printf("ERROR: malloc failed for %ld floats\n", (long)read_elems); return 1; }
    
    int n_read = gguf_read_tensor_f32(ctx, t, all_bos, read_elems);
    if (n_read <= 0) { 
        // Try with smaller buffer - read token by token
        printf("Direct read failed (%d), trying incremental...\n", n_read);
        free(all_bos);
        
        // Read first token (index 0)
        float *tok0 = (float *)malloc(D * sizeof(float));
        gguf_read_tensor_f32(ctx, t, tok0, D);
        printf("Token 0 first 10: ");
        for (int i = 0; i < 10; i++) printf("%.4f ", tok0[i]);
        printf("\n");
        free(tok0);
        
        // Read token 248044 directly via seeking to correct data offset
        char name[256];
        // We need to read the raw quantized data and dequantize
        printf("Using direct dequant...\n");
        
        // Alternative: read from the saved embeddings file
        FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (!f) { printf("ERROR: no embedding file\n"); return 1; }
        fseek(f, (long)BOS_ID * D * sizeof(float), SEEK_SET);
        fread(bos, sizeof(float), D, f);
        fclose(f);
        
        float mean = 0, var = 0;
        for (int i = 0; i < D; i++) mean += bos[i];
        mean /= D;
        for (int i = 0; i < D; i++) var += (bos[i]-mean)*(bos[i]-mean);
        var = sqrtf(var/D);
        printf("BOS emb from file: mean=%.6f std=%.6f\n", mean, var);
        printf("First 10: ");
        for (int i = 0; i < 10; i++) printf("%.4f ", bos[i]);
        printf("\n");
    } else {
        float *bos_vec = all_bos + BOS_ID * D;
        memcpy(bos, bos_vec, D * sizeof(float));
        float mean = 0, var = 0;
        for (int i = 0; i < D; i++) mean += bos[i];
        mean /= D;
        for (int i = 0; i < D; i++) var += (bos[i]-mean)*(bos[i]-mean);
        var = sqrtf(var/D);
        printf("BOS emb from GGUF: mean=%.6f std=%.6f\n", mean, var);
        printf("First 10: ");
        for (int i = 0; i < 10; i++) printf("%.4f ", bos[i]);
        printf("\n");
    }
    
    gguf_close(ctx);
    free(bos); if (all_bos) free(all_bos);
    printf("Done\n");
    return 0;
}
