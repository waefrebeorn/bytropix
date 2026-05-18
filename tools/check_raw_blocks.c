#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    // Instead of dequantizing, let's check the RAW Q4_K block data
    FILE *f = fopen("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
    // Find output.weight header by reading GGUF
    fseek(f, 4, SEEK_SET); // skip magic
    uint32_t version; fread(&version, 4, 1, f);
    uint64_t n_tensors; fread(&n_tensors, 8, 1, f);
    uint64_t n_kv; fread(&n_kv, 8, 1, f);
    
    // Skip KV
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen; fread(&klen, 8, 1, f); fseek(f, klen, SEEK_CUR);
        uint32_t vtype; fread(&vtype, 4, 1, f);
        switch (vtype) {
            case 0: fseek(f, 4, SEEK_CUR); break;
            case 1: fseek(f, 1, SEEK_CUR); break;
            case 2: fseek(f, 1, SEEK_CUR); break;
            case 3: fseek(f, 8, SEEK_CUR); break;
            case 4: fseek(f, 4, SEEK_CUR); break;
            case 5: fseek(f, 8, SEEK_CUR); break;
            case 6: case 7: case 8: { uint64_t al; fread(&al, 8, 1, f); fseek(f, al, SEEK_CUR); break; }
            default: { uint64_t al; fread(&al, 8, 1, f); fseek(f, al, SEEK_CUR); break; }
        }
    }
    
    // Find output.weight
    uint64_t out_offset = 0, out_size = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint32_t ndims; fread(&ndims, 4, 1, f);
        uint64_t nlen; fread(&nlen, 8, 1, f);
        char name[256]; fread(name, 1, nlen, f); name[nlen] = 0;
        uint64_t dim0, dim1; fread(&dim0, 8, 1, f); fread(&dim1, 8, 1, f);
        if (ndims > 2) { uint64_t d; fread(&d, 8, 1, f); }
        uint32_t gtype; fread(&gtype, 4, 1, f);
        fread(&out_offset, 8, 1, f);
        if (strcmp(name, "output.weight") == 0) {
            printf("output.weight at file offset %lu (data_blob_offset=10990048)\n", out_offset);
            printf("  dims=%lu x %lu, type=%u\n", dim0, dim1, gtype);
            break;
        }
    }
    
    // The actual data start in the file
    uint64_t data_pos = 10990048 + out_offset;
    printf("Data file position: %lu\n", data_pos);
    fseek(f, data_pos, SEEK_SET);
    
    // Read block 0 (176 bytes)
    uint8_t blk0[176]; fread(blk0, 1, 176, f);
    uint16_t d0_bits; memcpy(&d0_bits, blk0, 2);
    uint16_t dmin0_bits; memcpy(&dmin0_bits, blk0+2, 2);
    printf("Block 0: d_bits=%u dmin_bits=%u\n", d0_bits, dmin0_bits);
    
    // Read block 1 (176 bytes)  
    fseek(f, data_pos + 176, SEEK_SET);
    uint8_t blk1[176]; fread(blk1, 1, 176, f);
    uint16_t d1_bits; memcpy(&d1_bits, blk1, 2);
    uint16_t dmin1_bits; memcpy(&dmin1_bits, blk1+2, 2);
    printf("Block 1: d_bits=%u dmin_bits=%u\n", d1_bits, dmin1_bits);
    
    // Read block 4 (at 176*4 = 704)
    fseek(f, data_pos + 704, SEEK_SET);
    uint8_t blk4[176]; fread(blk4, 1, 176, f);
    uint16_t d4_bits; memcpy(&d4_bits, blk4, 2);
    uint16_t dmin4_bits; memcpy(&dmin4_bits, blk4+2, 2);
    printf("Block 4: d_bits=%u dmin_bits=%u\n", d4_bits, dmin4_bits);
    
    // Read block 1000
    fseek(f, data_pos + 1000*176, SEEK_SET);
    uint8_t blk1000[176]; fread(blk1000, 1, 176, f);
    uint16_t d1000_bits; memcpy(&d1000_bits, blk1000, 2);
    uint16_t dmin1000_bits; memcpy(&dmin1000_bits, blk1000+2, 2);
    printf("Block 1000: d_bits=%u dmin_bits=%u\n", d1000_bits, dmin1000_bits);
    
    fclose(f);
    return 0;
}
