#include "gguf_reader.h"
#include <stdio.h>
#include <string.h>

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { fprintf(stderr, "FAIL\n"); return 1; }
    
    printf("GGUF v%u, %lld tensors, %lld KV pairs\n", 
           ctx->version, (long long)ctx->n_tensors, (long long)ctx->n_kv);
    
    // Close and re-read from scratch to dump KV
    gguf_close(ctx);
    
    // Just use a simple raw read
    FILE *f = fopen("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
    if (!f) return 1;
    
    uint32_t magic;
    fread(&magic, 4, 1, f);
    uint32_t version;
    fread(&version, 4, 1, f);
    uint64_t n_tensors_, n_kv;
    fread(&n_tensors_, 8, 1, f);
    fread(&n_kv, 8, 1, f);
    
    for (uint64_t k = 0; k < n_kv; k++) {
        uint32_t key_len;
        fread(&key_len, 4, 1, f);
        char key[256];
        fread(key, 1, key_len, f);
        key[key_len] = '\0';
        
        uint32_t val_type;
        fread(&val_type, 4, 1, f);
        
        printf("  %s = ", key);
        
        switch (val_type) {
            case 0: { uint8_t v; fread(&v,1,1,f); printf("%u", v); break; }
            case 1: { int8_t v; fread(&v,1,1,f); printf("%d", v); break; }
            case 4: { uint32_t v; fread(&v,4,1,f); printf("%u", v); break; }
            case 5: { int32_t v; fread(&v,4,1,f); printf("%d", v); break; }
            case 6: { float v; fread(&v,4,1,f); printf("%f", v); break; }
            case 7: { bool v; fread(&v,1,1,f); printf("%s", v?"true":"false"); break; }
            case 8: { uint64_t sl; fread(&sl,8,1,f); char buf[65536]; fread(buf,1,sl,f); buf[sl]='\0'; printf("%s", buf); break; }
            case 9: { 
                uint32_t at; fread(&at,4,1,f); 
                uint64_t al; fread(&al,8,1,f); 
                printf("[arr:");
                for (uint64_t ai = 0; ai < al && ai < 12; ai++) {
                    if (at == 8) { // string array
                        uint64_t sl; fread(&sl,8,1,f); 
                        char buf[4096]; fread(buf,1,sl,f); buf[sl]='\0'; 
                        printf("%s ", buf);
                    } else if (at == 5) { // int32 array
                        int32_t v; fread(&v,4,1,f); printf("%d ", v);
                    } else if (at == 6) { // float32 array
                        float v; fread(&v,4,1,f); printf("%f ", v);
                    } else {
                        printf("(type%d) ", at);
                        fseek(f, 4, SEEK_CUR);
                    }
                }
                printf("]"); break; }
            case 10: { uint64_t v; fread(&v,8,1,f); printf("%llu", (unsigned long long)v); break; }
            case 11: { int64_t v; fread(&v,8,1,f); printf("%lld", (long long)v); break; }
            case 12: { double v; fread(&v,8,1,f); printf("%f", v); break; }
            default: printf("type=%u", val_type); fseek(f, 4, SEEK_CUR);
        }
        printf("\n");
    }
    
    fclose(f);
    return 0;
}
