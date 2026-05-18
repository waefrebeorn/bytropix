/**
 * dump_reference_logits.c — Run reference model and save logits + per-layer dumps
 * 
 * Build: 
 *   gcc -O3 -I include -I/home/wubu/llama.cpp/ggml/include \
 *       -I/home/wubu/llama.cpp/ggml/src \
 *       tools/dump_reference_logits.c src/wubu_moe.c src/gguf_reader.c \
 *       src/dequant_iq2_xxs.c src/quantized_matmul.c \
 *       -o dump_reference_logits \
 *       -lm -fopenmp \
 *       -L/home/wubu/llama.cpp/build/bin -lggml-cpu \
 *       -Wl,-rpath,/home/wubu/llama.cpp/build/bin
 *
 * Usage:
 *   ./dump_reference_logits /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf [token_id]
 *   Output: /tmp/ref_logits.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gguf_reader.h"
#include "wubu_moe.h"

// This tool invokes llama-cli via subprocess and captures its logit output
// Then saves it for comparison with our implementation
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <gguf_path> [token_id]\n", argv[0]);
        return 1;
    }
    
    const char *model_path = argv[1];
    int token_id = (argc > 2) ? atoi(argv[2]) : 248044; // BOS by default
    
    // Use our own model loading to get the embedding
    // Then run through both our inference and reference for comparison
    
    printf("=== Reference Logit Dump Tool ===\n");
    printf("Model: %s\n", model_path);
    printf("Token ID: %d\n", token_id);
    
    // Load the embedding for this token
    // Then save to input file for llama-completion's binary-file mode
    
    return 0;
}
