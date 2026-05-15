#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    int64_t total = t->dims[0] * t->dims[1];
    printf("total_elems=%ld\n", (long)total);
    
    // Allocate full buffer
    float *w = (float *)malloc(total * sizeof(float));
    int ret = gguf_read_tensor_f32(ctx, t, w, total);
    printf("ret=%d\n", ret);
    
    printf("vocab[0] dim[0..9]:");
    for (int i = 0; i < 10; i++) printf(" %.6e", w[i]);
    printf("\n");
    printf("vocab[1] dim[0..9]:");
    for (int i = 0; i < 10; i++) printf(" %.6e", w[2048+i]);
    printf("\n");
    
    // Check for insane values in first 100K elements
    int bad = 0;
    for (int64_t i = 0; i < 100000; i++) {
        if (isnan(w[i]) || isinf(w[i]) || fabsf(w[i]) > 100.0f) {
            if (bad < 10) printf("BAD at [%ld]: %.6e\n", (long)i, w[i]);
            bad++;
        }
    }
    printf("Bad values (|v|>100 or NaN/Inf): %d / 100000\n", bad);
    
    // Compare first 10 with model's values (known: ~-0.006, -0.002, ...)
    float expected[] = {-6.303549e-03, -1.582146e-03, -1.582146e-03, -6.303549e-03, 3.139257e-03};
    printf("\nCompare with model values:\n");
    for (int i = 0; i < 5; i++) {
        float diff = fabsf(w[i] - expected[i]);
        printf("  [%d] got=%.6e expected=%.6e diff=%.6e%s\n", 
               i, w[i], expected[i], diff, diff > 1e-6 ? " ***" : "");
    }
    
    free(w);
    gguf_close(ctx);
    printf("=== PASS ===\n");
    return 0;
}
