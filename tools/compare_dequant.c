/**
 * Compare Q5_K dequantization between our gguf_reader and llama.cpp.
 * Both should produce identical float values from the same quantized data.
 */
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Forward declare our ggml types (from gguf_reader.h)
enum ggml_type {
    GGML_TYPE_F32    = 0,
    GGML_TYPE_Q5_K   = 13,
};

// Our dequant functions
void dequantize_q5_K_row(const uint8_t *data, float *output, int64_t n_elems);

int main() {
    // Open the GGUF with our reader to get raw quantized data
    // For Q5_K, each block encodes 256 floats
    // Block size is 176 bytes (from gguf_reader.c: Q5_K_BLOCK_SIZE)
    
    // Read a small portion of a Q5_K tensor from the GGUF file directly
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    // We'll read the raw data from the file at the correct offset
    // First, find the offset of blk.0.attn_qkv.weight
    // We already know from our reader: it's a Q5_K tensor at some offset
    
    // For simplicity, let's create random Q5_K data and test both dequantizers
    // The Q5_K block has: d(2) + dmin(2) + scales(12) + qh(32) + qs(128) = 176 bytes
    uint8_t raw_block[176];
    // Fill with deterministic pattern
    for (int i = 0; i < 176; i++) raw_block[i] = (uint8_t)(i * 17 + 42);
    
    // Dequantize using our function
    float our_out[256];
    memset(our_out, 0, sizeof(our_out));
    dequantize_q5_K_row(raw_block, our_out, 256);
    
    // Now use llama.cpp's ggml dequant
    // In llama.cpp, the dequant function is accessible via the ggml API
    // ggml_type_name or ggml_dequantize could be used
    
    printf("Our Q5_K dequant first 10 values:\n");
    for (int i = 0; i < 10; i++) {
        printf("  [%d] = %.10f\n", i, our_out[i]);
    }
    
    // Also check: does llama.cpp have ggml_dequantize_row_q5_K or similar?
    // In newer llama.cpp, this is internal to ggml
    
    printf("\nTo compare: we need to call llama.cpp's ggml_dequantize_row_q5_K\n");
    printf("This function is in ggml-quants.h/cpp\n");
    
    return 0;
}
