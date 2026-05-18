#include "gguf_reader.h"
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if(!ctx) return 1;
    
    printf("data_blob_offset = %llu\n", (unsigned long long)ctx->data_blob_offset);
    printf("Has data_blob: %s\n", ctx->data_blob ? "yes" : "no");
    
    gguf_buffer_data(ctx);
    
    char mn[256];
    snprintf(mn,sizeof(mn),"blk.0.ffn_gate_exps.weight");
    gguf_tensor_info *t = gguf_find_tensor(ctx, mn);
    if(!t) return 1;
    
    uint64_t tensor_pos = ctx->data_blob_offset + t->data_offset;
    printf("data_offset = %llu\n", (unsigned long long)t->data_offset);
    printf("tensor_pos (file) = %llu\n", (unsigned long long)tensor_pos);
    
    // Compare: read first 20 bytes from file vs data_blob
    uint8_t from_file[100], from_blob[100];
    
    // From file
    fseek(ctx->file, tensor_pos, SEEK_SET);
    fread(from_file, 1, 100, ctx->file);
    
    // From data_blob
    memcpy(from_blob, ctx->data_blob + t->data_offset, 100);
    
    printf("\nComparison of first 100 bytes:\n");
    int diff_count = 0;
    for(int i=0;i<100;i++){
        if(from_file[i] != from_blob[i]) {
            if(diff_count < 10)
                printf("  byte %d: FILE=0x%02x BLOB=0x%02x\n", i, from_file[i], from_blob[i]);
            diff_count++;
        }
    }
    printf("Total differences: %d\n", diff_count);
    
    // Now compare at offset raw_one
    int64_t raw_one = gguf_raw_size(t->ggml_type, (int64_t)D_MODEL * D_FF);
    
    fseek(ctx->file, tensor_pos + raw_one, SEEK_SET);
    fread(from_file, 1, 100, ctx->file);
    memcpy(from_blob, ctx->data_blob + t->data_offset + raw_one, 100);
    
    printf("\nComparison at offset %lld (expert 1 boundary):\n", (long long)raw_one);
    diff_count = 0;
    for(int i=0;i<100;i++){
        if(from_file[i] != from_blob[i]) {
            if(diff_count < 10)
                printf("  byte %d: FILE=0x%02x BLOB=0x%02x\n", i, from_file[i], from_blob[i]);
            diff_count++;
        }
    }
    printf("Total differences: %d\n", diff_count);
    
    // Print first 20 bytes from both sources
    printf("\nFILE first 20 at tensor_pos+raw_one: ");
    for(int i=0;i<20;i++) printf("%02x ", from_file[i]);
    printf("\n");
    printf("BLOB first 20 at data_offset+raw_one: ");
    for(int i=0;i<20;i++) printf("%02x ", from_blob[i]);
    printf("\n");
    
    // What does the FILE show at tensor_pos?
    fseek(ctx->file, tensor_pos, SEEK_SET);
    fread(from_file, 1, 20, ctx->file);
    printf("\nFILE at tensor_pos: ");
    for(int i=0;i<20;i++) printf("%02x ", from_file[i]);
    printf("\n");
    
    gguf_close(ctx);
    return 0;
}
