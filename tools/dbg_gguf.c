#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    printf("Opening %s\n", path);
    fflush(stdout);
    
    FILE *f = fopen(path, "rb");
    if (!f) { printf("FAIL open\n"); return 1; }
    printf("Opened OK\n"); fflush(stdout);
    
    uint32_t magic;
    fread(&magic, 4, 1, f);
    printf("Magic: 0x%x\n", magic); fflush(stdout);
    
    uint32_t version;
    fread(&version, 4, 1, f);
    printf("Version: %u\n", version); fflush(stdout);
    
    uint64_t n_tensors, n_kv;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);
    printf("Tensors: %lu, KV: %lu\n", n_tensors, n_kv); fflush(stdout);
    
    // Try reading first K-V pair
    for (uint64_t i = 0; i < n_kv && i < 5; i++) {
        uint64_t klen;
        if (fread(&klen, 8, 1, f) != 1) { printf("FAIL read klen at %lu\n", i); break; }
        char *kname = malloc(klen + 1);
        if (fread(kname, 1, klen, f) != klen) { printf("FAIL read key\n"); break; }
        kname[klen] = '\0';
        uint32_t kt;
        fread(&kt, 4, 1, f);
        printf("KV[%lu]: '%s' type=%u\n", i, kname, kt);
        
        if (strcmp(kname, "tokenizer.ggml.tokens") == 0) {
            // Read array header
            uint32_t arr_type; fread(&arr_type, 4, 1, f);
            uint64_t arr_len; fread(&arr_len, 8, 1, f);
            printf("  array type=%u len=%lu\n", arr_type, arr_len);
            // Read first token
            uint64_t slen; fread(&slen, 8, 1, f);
            printf("  first token len=%lu\n", slen);
            printf("  first token starts with: '%s'...\n", slen > 0 ? "skipped" : "empty");
            fseek(f, slen, SEEK_CUR);  // skip first token data
            // skip the rest
            for (uint64_t j = 1; j < arr_len; j++) {
                uint64_t sj; fread(&sj, 8, 1, f);
                fseek(f, sj, SEEK_CUR);
            }
        } else {
            // Skip based on type
            if (kt == 0) { uint64_t sl; fread(&sl, 8, 1, f); fseek(f, sl, SEEK_CUR); }
            else if (kt == 3) {
                uint32_t at; fread(&at, 4, 1, f);
                uint64_t al; fread(&al, 8, 1, f);
                for (uint64_t j = 0; j < al; j++) {
                    if (at == 0) { uint64_t sl2; fread(&sl2, 8, 1, f); fseek(f, sl2, SEEK_CUR); }
                    else if (at == 4) fseek(f, 4, SEEK_CUR);
                    else if (at == 5) fseek(f, 4, SEEK_CUR);
                    else if (at == 8) fseek(f, 8, SEEK_CUR);
                    else fseek(f, 8, SEEK_CUR);
                }
            }
            else if (kt == 4) fseek(f, 4, SEEK_CUR);
            else if (kt == 5) fseek(f, 4, SEEK_CUR);
            else if (kt == 8) fseek(f, 8, SEEK_CUR);
            else fseek(f, 8, SEEK_CUR);
        }
        free(kname);
    }
    
    printf("DONE\n");
    fclose(f);
    return 0;
}
