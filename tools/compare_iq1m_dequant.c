// compare_iq1m_dequant.c — Compare bytropix vs llama.cpp IQ1_M dequant
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK_K 256

int main(int argc, char **argv) {
    // Open model, read a single IQ1_M block, dequant with both libs
    const char *path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ1_M.gguf";
    
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 1; }
    
    // Skip GGUF header to find tensor data
    // GGUF3 header: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8)
    // Then metadata KV pairs, then tensor infos, then aligned data
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    uint8_t *blob = (uint8_t *)malloc(fsize);
    fread(blob, 1, fsize, f);
    fclose(f);
    
    // Parse GGUF header
    uint32_t magic; fread(&magic, 4, 1, f);
    (void)magic;
    
    // We already read the whole file, just use the blob directly
    uint32_t *hdr = (uint32_t *)blob;
    printf("magic=0x%x version=%u\n", hdr[0], hdr[1]);
    
    // Just read a block of IQ1_M data directly from a known offset
    // For a real test we'd parse tensor info, but for now let's read from
    // the file at a known IQ1_M tensor's data offset
    
    printf("File size: %ld bytes\n", fsize);
    printf("Need to parse tensor info to find IQ1_M weights\n");
    
    // We can also just write this test using our gguf_reader library
    printf("Use gguf_reader-based test instead\n");
    
    free(blob);
    return 0;
}
