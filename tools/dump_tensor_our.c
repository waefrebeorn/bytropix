/**
 * dump_tensor_our.c — Dump a specific tensor dequantized by our gguf_reader.
 * Build: gcc -O2 -I include -o dump_tensor_our tools/dump_tensor_our.c src/gguf_reader.o -lm
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s model.gguf tensor_name output.bin\n", argv[0]);
        return 1;
    }

    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) {
        fprintf(stderr, "Failed to open %s\n", argv[1]);
        return 1;
    }

    // Find tensor by name
    gguf_tensor_info *target = NULL;
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, argv[2]) == 0) {
            target = &ctx->tensors[i];
            break;
        }
    }

    if (!target) {
        fprintf(stderr, "Tensor '%s' not found\n", argv[2]);
        fprintf(stderr, "Available tensors (first 20):\n");
        for (int i = 0; i < ctx->n_tensors && i < 20; i++)
            fprintf(stderr, "  %s [type=%d]\n", ctx->tensors[i].name, ctx->tensors[i].ggml_type);
        gguf_close(ctx);
        return 1;
    }

    int64_t n_elems = 1;
    for (int d = 0; d < target->n_dims; d++)
        n_elems *= target->dims[d];

    int64_t raw_size = gguf_raw_size(target->ggml_type, n_elems);
    uint64_t tensor_abs_offset = ctx->data_blob_offset + target->data_offset;

    fprintf(stderr, "Tensor: %s\n", target->name);
    fprintf(stderr, "  type=%d n_dims=%d n_elems=%ld\n",
            target->ggml_type, target->n_dims, n_elems);
    fprintf(stderr, "  dims: ");
    for (int d = 0; d < target->n_dims; d++)
        fprintf(stderr, "%ld ", target->dims[d]);
    fprintf(stderr, "\n");
    fprintf(stderr, "  data_offset=%lu raw_size=%ld tensor_abs_offset=%lu\n",
            target->data_offset, raw_size, tensor_abs_offset);
    fprintf(stderr, "  data_blob_offset=%lu\n", ctx->data_blob_offset);

    // Read raw bytes from position 24576*176 = 4325376 (start of 16384th block)
    // Actually, let me check the raw data at the boundary between blocks 255-256
    int64_t boundary = 256 * 176;  // start of block 256 raw data
    fprintf(stderr, "\n  Block 256 starts at raw offset %ld in tensor data\n", boundary);
    
    // Read raw bytes at boundaries
    uint8_t raw_at_block255[16];  // first 16 bytes of block 255
    fseek(ctx->file, tensor_abs_offset + (255 * 176), SEEK_SET);
    fread(raw_at_block255, 1, 16, ctx->file);
    fprintf(stderr, "\n  Block 255 first 16 raw bytes: ");
    for (int i = 0; i < 16; i++) fprintf(stderr, "%02x ", raw_at_block255[i]);
    fprintf(stderr, "\n");
    
    uint8_t raw_at_block256[16];  // first 16 bytes of block 256  
    fseek(ctx->file, tensor_abs_offset + boundary, SEEK_SET);
    fread(raw_at_block256, 1, 16, ctx->file);
    fprintf(stderr, "  Block 256 first 16 raw bytes: ");
    for (int i = 0; i < 16; i++) fprintf(stderr, "%02x ", raw_at_block256[i]);
    fprintf(stderr, "\n");

    uint8_t raw_at_block257[16];
    fseek(ctx->file, tensor_abs_offset + (257 * 176), SEEK_SET);
    fread(raw_at_block257, 1, 16, ctx->file);
    fprintf(stderr, "  Block 257 first 16 raw bytes: ");
    for (int i = 0; i < 16; i++) fprintf(stderr, "%02x ", raw_at_block257[i]);
    fprintf(stderr, "\n");

    // Also check: does the tensor data region end before block 256?
    fseek(ctx->file, 0, SEEK_END);
    long file_size = ftell(ctx->file);
    fprintf(stderr, "\n  File size: %ld\n", file_size);
    fprintf(stderr, "  Tensor data end: %lu (within file: %s)\n",
            tensor_abs_offset + raw_size,
            (tensor_abs_offset + raw_size <= (uint64_t)file_size) ? "YES" : "NO");
    
    // Check next tensor's data_offset (if this isn't the last)
    for (int i = 0; i < ctx->n_tensors; i++) {
        if (ctx->tensors[i].data_offset > target->data_offset &&
            strcmp(ctx->tensors[i].name, target->name) != 0) {
            // This is a later tensor
            uint64_t next_offset = ctx->data_blob_offset + ctx->tensors[i].data_offset;
            fprintf(stderr, "\n  Next tensor: %s at offset %lu\n", 
                    ctx->tensors[i].name, next_offset);
            fprintf(stderr, "  Gap: %lu bytes\n", next_offset - tensor_abs_offset);
            fprintf(stderr, "  Our raw_size says: %ld bytes\n", raw_size);
            break;
        }
    }

    // Dequantize and write
    float *buf = (float *)malloc(n_elems * sizeof(float));
    if (!buf) { gguf_close(ctx); return 1; }

    int n_read = gguf_read_tensor_f32(ctx, target, buf, n_elems);
    fprintf(stderr, "\n  Dequantized %d/%ld elements\n", n_read, n_elems);

    // Write but also dump raw size info
    FILE *f = fopen(argv[3], "wb");
    if (!f) { free(buf); gguf_close(ctx); return 1; }
    fwrite(buf, sizeof(float), n_read, f);
    fclose(f);

    float mn=buf[0], mx=buf[0];
    double sum=0;
    for(int i=0; i<n_read; i++){
        sum+=buf[i];
        if(buf[i]<mn) mn=buf[i];
        if(buf[i]>mx) mx=buf[i];
    }
    fprintf(stderr, "Stats: mean=%.4f max=%.4f min=%.4f |v|>10=%d/%d\n",
            (float)(sum/n_read), mx, mn, 0, n_read);
    fprintf(stderr, "First 8: ");
    for(int i=0;i<8&&i<n_read;i++) fprintf(stderr,"%+.6f ",buf[i]);
    fprintf(stderr,"\n");

    free(buf);
    gguf_close(ctx);
    return 0;
}
