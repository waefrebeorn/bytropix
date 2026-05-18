#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path1 = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *path2 = argc > 2 ? argv[2] : "/models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf";
    
    gguf_ctx *ctx1 = gguf_open(path1);
    gguf_ctx *ctx2 = gguf_open(path2);
    if (!ctx1 || !ctx2) { return 1; }
    
    printf("Model 1: %s\n", path1);
    printf("Model 2: %s\n", path2);
    printf("Tensors: %ld vs %ld\n", (long)ctx1->n_tensors, (long)ctx2->n_tensors);
    printf("\nTensors in Model 2 NOT in Model 1:\n");
    
    for (int i2 = 0; i2 < (int)ctx2->n_tensors; i2++) {
        int found = 0;
        for (int i1 = 0; i1 < (int)ctx1->n_tensors; i1++) {
            if (strcmp(ctx2->tensors[i2].name, ctx1->tensors[i1].name) == 0) {
                found = 1; break;
            }
        }
        if (!found) {
            printf("  %s\n", ctx2->tensors[i2].name);
        }
    }
    
    gguf_close(ctx1);
    gguf_close(ctx2);
    return 0;
}
