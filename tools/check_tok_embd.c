/**
 * check_tok_embd.c — Extract token_embd.weight from GGUF and compare with file
 * Build: gcc -O2 -I include -o check_tok_embd tools/check_tok_embd.c src/gguf_reader.o -lm
 * Usage: ./check_tok_embd [token_id]
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define D_MODEL 2048

int main(int argc, char **argv) {
    const char *model_path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int token_id = 248044;
    if (argc > 1) token_id = atoi(argv[1]);

    // Open GGUF
    gguf_ctx *ctx = gguf_open(model_path);
    if (!ctx) { fprintf(stderr, "Failed to open GGUF\n"); return 1; }

    // Buffer the full data blob
    if (!gguf_buffer_data(ctx)) {
        fprintf(stderr, "Failed to buffer data\n");
        gguf_close(ctx);
        return 1;
    }

    // Find token_embd.weight tensor
    gguf_tensor_info *t_emb = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t_emb) { fprintf(stderr, "token_embd.weight not found\n"); gguf_close(ctx); return 1; }
    fprintf(stderr, "token_embd.weight: type=%d, dims=[%lld, %lld]\n", 
            t_emb->ggml_type, (long long)t_emb->dims[0], (long long)t_emb->dims[1]);

    // Read the full tensor (all tokens) — but only for BOS
    // Actually token_embd.weight is [n_embd, n_vocab] = [2048, 248320]
    // We need to read from offset = 0 (all data) but the tensor is quantized (Q5_K)
    // so gguf_read_tensor_f32 will dequantize — this reads the WHOLE tensor which is huge!
    // Instead, let's just read the first few tokens and verify manually.
    
    // Actually let's just read a small slice: first 10 tokens × D_MODEL
    int64_t n_test = 10 * D_MODEL;
    float *buf = (float *)malloc(n_test * sizeof(float));
    int64_t n_read = gguf_read_tensor_f32(ctx, t_emb, buf, n_test);
    if (n_read <= 0) {
        fprintf(stderr, "Failed to read tensor data (n_read=%lld)\n", (long long)n_read);
        free(buf);
        gguf_close(ctx);
        return 1;
    }
    fprintf(stderr, "Read %lld floats from tensor\n", (long long)n_read);

    // Compare token_id with the extracted file
    const char *file_path = "data/qwen36_embeddings_c.bin.raw";
    FILE *f = fopen(file_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", file_path); free(buf); gguf_close(ctx); return 1; }

    fprintf(stderr, "\nVerifying token=%d from GGUF vs file:\n", token_id);
    double dot = 0, n1 = 0, n2 = 0, max_diff = 0;
    int max_diff_idx = 0;
    for (int i = 0; i < D_MODEL; i++) {
        // From GGUF: the tensor is [n_embd, n_vocab], so element at (i, token_id)
        // = buf[token_id * D_MODEL + i]
        float gguf_val = buf[token_id * D_MODEL + i];
        
        // From file: seek to token_id * D_MODEL
        // But we need to read from file
        fseek(f, (long long)token_id * D_MODEL * sizeof(float), SEEK_SET);
        float file_val;
        fread(&file_val, sizeof(float), 1, f);
        // Rewind each time is slow but for D_MODEL=2048 it's fine
        
        double diff = fabs(gguf_val - file_val);
        if (diff > max_diff) { max_diff = diff; max_diff_idx = i; }
        dot += (double)gguf_val * file_val;
        n1 += (double)gguf_val * gguf_val;
        n2 += (double)file_val * file_val;
    }
    fclose(f);
    
    double cos_sim = dot / (sqrt(n1) * sqrt(n2) + 1e-30);
    fprintf(stderr, "  Cos-sim: %.15f\n", cos_sim);
    fprintf(stderr, "  Max diff: %.15f at idx %d\n", max_diff, max_diff_idx);
    fprintf(stderr, "  GGUF std=%.6f, File std=%.6f\n",
            sqrt(n1/D_MODEL), sqrt(n2/D_MODEL));

    // Also check if dim order is correct: maybe tensor is [n_vocab, n_embd] not [n_embd, n_vocab]
    // Try transposed layout: element at (token_id, i) instead
    fprintf(stderr, "\nTrying alternative layout (token as dim[0]):\n");
    dot = 0; n1 = 0; n2 = 0; max_diff = 0;
    f = fopen(file_path, "rb");
    if (f) {
        fseek(f, (long long)token_id * D_MODEL * sizeof(float), SEEK_SET);
        for (int i = 0; i < D_MODEL; i++) {
            // If tensor is [n_vocab, n_embd], element at (token_id, i) = buf[token_id * D_MODEL + i]
            // Same as above because it's 2D, [d0, d1] with d0 being the last dim in GGUF
            // Actually in GGUF, dims[0] is the innermost dimension
            // token_embd.weight: gguf_read_tensor_f32 reads as row-major
            // If dims = [2048, 248320], then buf[token_id * 2048 + i] is correct
            // If dims = [248320, 2048], then buf[i * 248320 + token_id] is correct
            float alt_val = buf[i * 248320 + token_id]; // VERY slow, but let's see first few
            if (i < 5) {
                float file_val;
                fread(&file_val, sizeof(float), 1, f);
                fprintf(stderr, "  alt[%d]: GGUF=%.10f File=%.10f\n", i, alt_val, file_val);
            }
        }
        fclose(f);
    }

    free(buf);
    gguf_close(ctx);
    fprintf(stderr, "\nCheck complete.\n");
    return 0;
}
