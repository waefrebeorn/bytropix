/* Compare output.weight dequant between our reader and llama.cpp */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gguf_reader.h"
#include "llama.h"

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    // ---- Our gguf_reader ----
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    if (!t) { printf("ERROR: no output.weight\n"); return 1; }
    printf("output.weight: dims=[%ld,%ld] type=%d\n", (long)t->dims[0], (long)t->dims[1], t->ggml_type);
    
    // Read first 1000 floats
    int N = 1000;
    float *our = (float*)malloc(N * sizeof(float));
    // Since gguf_read_tensor_f32 requires n_elems <= max_elems, we need to read all or none
    // But the tensor has 508M elements... we can't read just 1000.
    // Let's read just the first token worth (2048 floats)
    int D = 2048;
    float *our_full = (float*)malloc((int64_t)D * sizeof(float));
    // Actually the function reads from the start, can't skip.
    // We have to read the whole tensor and take first D.
    // Too big (508M floats = 2GB). 
    // Alternative: modify gguf_reader to support partial reads... or
    
    // Just read the raw data and manually dequantize first block
    // output.weight is Q4_K type=12
    // Q4_K block = 256 elements, block size = 144 bytes
    // First block offset = tensor->data_offset
    const uint8_t *raw = (const uint8_t*)ctx->data_blob + t->data_offset;
    
    // Read first Q4_K block (144 bytes) and dequantize to 256 floats
    float *our_block = (float*)malloc(256 * sizeof(float));
    dequantize_q4_K_row(raw, our_block, 256);
    
    printf("Our output.weight first 256 elements (first Q4_K block):\n");
    for (int i = 0; i < 10 && i < 256; i++) printf(" %.6f", our_block[i]);
    printf("\n");
    float m=0; for(int i=0;i<256;i++) m+=our_block[i];
    printf("mean=%.6f\n", m/256);
    
    gguf_close(ctx);
    
    // ---- llama.cpp ----
    // For llama.cpp, we need to get the output weight directly from the model
    // The llama.h API doesn't expose tensor access, but we can use gguf context
    
    // Actually, let's use the gguf library from llama.cpp
    // libllama.so is linked. Let's use its gguf API
    // Include the gguf header...
    
    // Alternative: use Python with gguf library to dequant
    // For now, let's just verify our Q4_K dequant by comparing with a known-good reference
    
    // Save our block for external comparison
    FILE *f = fopen("/tmp/our_outw_block.bin", "wb");
    fwrite(our_block, sizeof(float), 256, f);
    fclose(f);
    printf("Saved first Q4_K block to /tmp/our_outw_block.bin\n");
    
    free(our_block); free(our_full);
    return 0;
}
