/**
 * dump_mmproj.c — Read and verify MMProj vision merger GGUF dimensions
 *
 * Reads /models/qwen3.6-35b-mmproj-F16.gguf and dumps all tensor names
 * and shapes, with verification of the known architecture from target map.
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/models/qwen3.6-35b-mmproj-F16.gguf";
    printf("=== MMProj Vision Merger — Dimension Verification ===\n");
    printf("File: %s\n\n", path);

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open %s\n", path); return 1; }

    printf("Tensors: %ld\n", (long)ctx->n_tensors);

    // Check all tensors
    int has_mm0 = 0, has_mm2 = 0, has_mm1 = 0;
    int n_vision_blocks = 0;
    int has_patch_embd = 0, has_pos_embd = 0, has_post_ln = 0;

    printf("\n--- All Tensors ---\n");
    for (int i = 0; i < (int)ctx->n_tensors; i++) {
        gguf_tensor_info *t = &ctx->tensors[i];
        
        // Print dims
        printf("  T[%3d] %s [", i, t->name);
        for (int d = 0; d < t->n_dims; d++) {
            printf("%ld%s", (long)t->dims[d], d < t->n_dims-1 ? "," : "");
        }
        printf("] type=%d\n", t->ggml_type);

        // Track key tensors
        if (strcmp(t->name, "mm.0.weight") == 0) {
            has_mm0 = 1;
            printf("    → %s [%ld,%ld]\n", 
                   (t->dims[0]==4608 && t->dims[1]==4608) ? "PASS" : "FAIL",
                   (long)t->dims[0], (long)t->dims[1]);
        }
        if (strcmp(t->name, "mm.2.weight") == 0) {
            has_mm2 = 1;
            printf("    → %s [%ld,%ld] (expected [4608,2048])\n",
                   (t->dims[0]==4608 && t->dims[1]==2048) ? "PASS" : "FAIL",
                   (long)t->dims[0], (long)t->dims[1]);
        }
        if (strncmp(t->name, "mm.1", 4) == 0) {
            has_mm1 = 1;
        }
        if (strcmp(t->name, "v.patch_embd.weight") == 0) {
            has_patch_embd = 1;
            printf("    → %s [%ld,%ld,%ld,%ld]\n",
                   (t->dims[0]==16 && t->dims[1]==16 && t->dims[2]==3 && t->dims[3]==1152) ? "PASS" : "FAIL",
                   (long)t->dims[0], (long)t->dims[1], (long)t->dims[2], (long)t->dims[3]);
        }
        if (strcmp(t->name, "v.position_embd.weight") == 0) {
            has_pos_embd = 1;
            printf("    → %s [%ld,%ld] (expected [1152,2304])\n",
                   (t->dims[0]==1152 && t->dims[1]==2304) ? "PASS" : "FAIL",
                   (long)t->dims[0], (long)t->dims[1]);
        }
        if (strcmp(t->name, "v.post_ln.weight") == 0) {
            has_post_ln = 1;
        }

        // Count vision blocks
        if (strstr(t->name, "v.blk.")) {
            int blk;
            if (sscanf(t->name, "v.blk.%d.", &blk) == 1) {
                if (blk + 1 > n_vision_blocks) n_vision_blocks = blk + 1;
            }
        }
    }

    printf("\n--- Verification ---\n");
    printf("  mm.0.weight [4608,4608]: %s\n", has_mm0 ? "FOUND ✓" : "MISSING ✗");
    printf("  mm.2.weight [4608,2048]: %s\n", has_mm2 ? "FOUND ✓" : "MISSING ✗");
    printf("  mm.1.* (should NOT exist): %s\n", has_mm1 ? "FOUND (UNEXPECTED) ✗" : "NOT FOUND ✓");
    printf("  Vision blocks: %d\n", n_vision_blocks);
    printf("  v.patch_embd.weight: %s\n", has_patch_embd ? "FOUND ✓" : "MISSING ✗");
    printf("  v.position_embd.weight: %s\n", has_pos_embd ? "FOUND ✓" : "MISSING ✗");
    printf("  v.post_ln.weight/bias: %s\n", has_post_ln ? "FOUND ✓" : "MISSING ✗");
    printf("  Total: %ld tensors (expected ~334)\n", (long)ctx->n_tensors);

    gguf_close(ctx);

    int pass = has_mm0 && has_mm2 && !has_mm1 && n_vision_blocks == 27 && has_patch_embd && has_pos_embd;
    printf("\n=== %s ===\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
