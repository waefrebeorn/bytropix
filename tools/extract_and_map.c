#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *gguf_path = "/mnt/wslg/distro/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *out_path = "/home/wubu/bytropix/data/qwen36_embeddings_c.bin";
    float R = 0.956f;  // Poincaré ball radius from analysis
    
    if (argc > 1) gguf_path = argv[1];
    if (argc > 2) out_path = argv[2];
    if (argc > 3) R = atof(argv[3]);
    
    printf("Opening GGUF: %s\n", gguf_path);
    
    gguf_ctx *ctx = gguf_open(gguf_path);
    if (!ctx) {
        fprintf(stderr, "Failed to open GGUF\n");
        return 1;
    }
    
    // Find token_embd.weight
    gguf_tensor_info *tensor = gguf_find_tensor(ctx, "token_embd.weight");
    if (!tensor) {
        fprintf(stderr, "token_embd.weight not found\n");
        gguf_close(ctx);
        return 1;
    }
    
    // Shape: [hidden, vocab] in GGUF column-major
    int64_t hidden = tensor->dims[0];
    int64_t vocab = tensor->dims[1];
    printf("Found: %s [%ld, %ld], type=%d\n", tensor->name, vocab, hidden, tensor->ggml_type);
    
    int64_t n_elems = hidden * vocab;
    printf("Total elements: %ld (%.2fM)\n", n_elems, n_elems / 1e6);
    
    // Allocate output buffer (2.03GB float32)
    float *embeddings = (float*)malloc(n_elems * sizeof(float));
    if (!embeddings) {
        fprintf(stderr, "Failed to allocate %ld bytes\n", n_elems * sizeof(float));
        gguf_close(ctx);
        return 1;
    }
    
    // Read and dequantize
    printf("Reading tensor data...\n");
    int n_read = gguf_read_tensor_f32(ctx, tensor, embeddings, n_elems);
    if (n_read != n_elems) {
        fprintf(stderr, "Only read %d / %ld elements\n", n_read, n_elems);
        free(embeddings);
        gguf_close(ctx);
        return 1;
    }
    
    printf("Read %d elements\n", n_read);
    
    // Compute stats on raw embeddings
    double mean = 0.0, std = 0.0, min_val = 1e30, max_val = -1e30;
    double norm_sum = 0.0, norm_sum_sq = 0.0, norm_min = 1e30, norm_max = -1e30;
    
    for (int64_t v = 0; v < vocab; v++) {
        float norm = 0.0;
        for (int64_t h = 0; h < hidden; h++) {
            float val = embeddings[v * hidden + h];
            mean += val;
            std += val * val;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            norm += val * val;
        }
        norm = sqrtf(norm);
        norm_sum += norm;
        norm_sum_sq += norm * norm;
        if (norm < norm_min) norm_min = norm;
        if (norm > norm_max) norm_max = norm;
    }
    
    mean /= (vocab * hidden);
    std = sqrt(std / (vocab * hidden) - mean * mean);
    norm_sum /= vocab;
    norm_sum_sq = sqrt(norm_sum_sq / vocab - norm_sum * norm_sum);
    
    printf("\nEmbedding stats:\n");
    printf("  value: mean=%.6f, std=%.6f, range=[%.6f, %.6f]\n", mean, std, min_val, max_val);
    printf("  norm:  mean=%.6f, std=%.6f, range=[%.6f, %.6f]\n", norm_sum, norm_sum_sq, norm_min, norm_max);
    
    // Apply Poincaré exp_map
    printf("\nMapping to Poincaré ball (R=%.4f)...\n", R);
    float *mapped = (float*)malloc(n_elems * sizeof(float));
    if (!mapped) {
        free(embeddings);
        gguf_close(ctx);
        return 1;
    }
    
    for (int64_t v = 0; v < vocab; v++) {
        wubu_exp_map(&embeddings[v * hidden], hidden, R, &mapped[v * hidden]);
    }
    
    // Stats on mapped embeddings
    double p_norm_sum = 0.0, p_norm_min = 1e30, p_norm_max = -1e30;
    for (int64_t v = 0; v < vocab; v++) {
        float norm = wubu_norm(&mapped[v * hidden], hidden);
        p_norm_sum += norm;
        if (norm < p_norm_min) p_norm_min = norm;
        if (norm > p_norm_max) p_norm_max = norm;
    }
    p_norm_sum /= vocab;
    
    printf("  Poincaré norms: mean=%.6f, range=[%.6f, %.6f]\n", p_norm_sum, p_norm_min, p_norm_max);
    
    // Save mapped embeddings
    printf("\nSaving mapped embeddings to %s...\n", out_path);
    FILE *out = fopen(out_path, "wb");
    if (!out) {
        fprintf(stderr, "Failed to open output\n");
        free(embeddings); free(mapped);
        gguf_close(ctx);
        return 1;
    }
    fwrite(mapped, sizeof(float), n_elems, out);
    fclose(out);
    
    // Also save raw embeddings
    char raw_out[512];
    snprintf(raw_out, sizeof(raw_out), "%s.raw", out_path);
    printf("Saving raw embeddings to %s...\n", raw_out);
    out = fopen(raw_out, "wb");
    if (out) {
        fwrite(embeddings, sizeof(float), n_elems, out);
        fclose(out);
    }
    
    // Save meta
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s.meta", out_path);
    printf("Saving meta to %s...\n", meta_path);
    out = fopen(meta_path, "w");
    if (out) {
        fprintf(out, "shape %ld %ld\n", vocab, hidden);
        fprintf(out, "source %s\n", gguf_path);
        fprintf(out, "tensor token_embd.weight\n");
        fprintf(out, "dtype float32\n");
        fprintf(out, "norm_mean %.6f\n", norm_sum);
        fprintf(out, "norm_std %.6f\n", norm_sum_sq);
        fprintf(out, "norm_min %.6f\n", norm_min);
        fprintf(out, "norm_max %.6f\n", norm_max);
        fprintf(out, "poincare_R %.4f\n", R);
        fprintf(out, "poincare_norm_mean %.6f\n", p_norm_sum);
        fprintf(out, "poincare_norm_min %.6f\n", p_norm_min);
        fprintf(out, "poincare_norm_max %.6f\n", p_norm_max);
        fclose(out);
    }
    
    free(embeddings);
    free(mapped);
    gguf_close(ctx);
    
    printf("\nDone! Files:\n");
    printf("  %s (mapped, %.2fGB)\n", out_path, n_elems * 4.0 / 1e9);
    printf("  %s (raw, %.2fGB)\n", raw_out, n_elems * 4.0 / 1e9);
    printf("  %s (meta)\n", meta_path);
    
    return 0;
}
