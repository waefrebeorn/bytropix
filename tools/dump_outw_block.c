/* Save first 144 bytes of output.weight (one Q4_K block) + dequant values */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "gguf_reader.h"

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    
    gguf_tensor_info *t = gguf_find_tensor(ctx, "output.weight");
    const uint8_t *data = (const uint8_t*)ctx->data_blob + t->data_offset;
    
    // Save first 4 Q4_K blocks for analysis (576 bytes = 1024 elements)
    FILE *f = fopen("/tmp/raw_outw_4blk.bin", "wb");
    fwrite(data, 1, 576, f);
    fclose(f);
    
    // Also save first 2048 floats (output token 0) by doing dequant ourselves
    // We need to dequant manually... let's just use gguf_read_tensor_f32
    int D = 2048;
    float *first_token = (float*)malloc(D * sizeof(float));
    // gguf_read_tensor_f32 reads from the START of the tensor
    // Since output.weight dims=[D_MODEL, VOCAB], first D_MODEL elements = output token 0
    // But gguf_read_tensor_f32 reads n_elems from the start
    // For Q4_K, reading 2048 elements = 8 blocks = 1152 bytes raw
    // This should work
    
    int n_read = gguf_read_tensor_f32(ctx, t, first_token, D);
    if (n_read > 0) {
        f = fopen("/tmp/our_outw_token0.bin", "wb");
        fwrite(first_token, sizeof(float), n_read, f);
        fclose(f);
        printf("Saved %d floats for output token 0\n", n_read);
        
        // Stats
        double m = 0, s = 0;
        for (int i = 0; i < n_read; i++) {
            m += first_token[i];
            s += first_token[i] * first_token[i];
        }
        printf("token0: mean=%.8f std=%.8f first=%.8f\n", m/n_read, sqrt(s/n_read - (m/n_read)*(m/n_read)), first_token[0]);
    }
    
    gguf_close(ctx);
    free(first_token);
    return 0;
}
