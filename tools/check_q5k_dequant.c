/* Cross-check: dequant one Q5_K block using ggml's API and compare with ours.
   Build: g++ -O2 -I /home/wubu/llama.cpp/ggml/include -o /tmp/chk_q5k tools/check_q5k_dequant.c \
          -L /home/wubu/llama.cpp/build/bin -lggml -lggml-base -lggml-cpu -lm -Wl,-rpath,/home/wubu/llama.cpp/build/bin */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "ggml.h"

int main() {
    // First, let's read the raw bytes of a Q5_K block from the GGUF file
    FILE *f = fopen("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
    if (!f) return 1;
    
    // Find the offset of token_embd.weight (first Q5_K tensor)
    // Just search for "token_embd.weight" in the file
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    
    // Read entire GGUF header to find tensor offsets
    // Or: just read the first Q5_K block we know exists
    // Let me use a simpler approach: read from a known position
    
    // The tensor data section starts after all metadata. Let me find it.
    // GGUF v3: header = magic(4) + version(4) + n_tensors(8) + n_kv(8) + metadata + tensor_infos
    // Then: padding to alignment, then tensor data
    
    // Let me just read at a known position near the end of file where tensor data is
    fclose(f);
    
    // Second approach: use llama API to get actual dequantized values
    // and compare with our bytropix values
    
    // Load llama model
    llama_backend_init();
    struct llama_model_params mlp = llama_model_default_params();
    mlp.n_gpu_layers = 0;
    mlp.use_mmap = false;
    struct llama_model *model = llama_model_load_from_file(
        "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", mlp);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }
    
    printf("Model loaded OK via llama\n");
    
    // The llama API doesn't expose raw weight tensors directly
    // Let me use ggml directly by reading the GGUF data
    
    // Third approach: read the raw bytes of a Q5_K block from a known location
    // and dequant using both our C code's method and ggml's method
    
    // I know our C code's Q5_K dequant is in gguf_reader.c
    // Let me test that function against ggml's internal dequant
    
    // Just read the raw file to get the first Q5_K block of output.weight
    // output.weight is Q4_K (type 12 in our enum), so that won't work
    // Let me find token_embd.weight which is Q5_K
    
    // Actually, let me just manually extract using the ggml dequant function
    struct ggml_init_params params = {
        .mem_size = 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context *ctx = ggml_init(params);
    
    // Create a Q5_K tensor and set its data
    struct ggml_tensor *t = ggml_new_tensor_1d(ctx, GGML_TYPE_Q5_K, 256);
    
    // Fill with test data from our file
    // Read the first block of token_embd for token 0
    f = fopen("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
    // Skip to the first tensor's data. We don't know exact offset...
    // Let me use a known position by finding the data start from GGUF headers
    fclose(f);
    
    // OK this is getting complicated. Let me use a different approach.
    // Let me skip ggml entirely and directly compare: our dequant of a Q5_K 
    // block vs ggml's dequant of the same block, by reading from known offsets.
    
    printf("Using Python dequant comparison instead...\n");
    
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
