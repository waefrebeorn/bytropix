#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "gguf_reader.h"

int main() {
    gguf_ctx *ctx = gguf_open("/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) { fprintf(stderr, "failed to open\n"); return 1; }
    
    for (int i = 0; i < ctx->n_kv; i++) {
        gguf_kv_t *kv = &ctx->kv[i];
        printf("  %s: type=%d ", kv->key, kv->type);
        
        switch (kv->type) {
            case GGUF_TYPE_UINT8:   printf("= %u\n", kv->value.uint8); break;
            case GGUF_TYPE_INT8:    printf("= %d\n", kv->value.int8); break;
            case GGUF_TYPE_UINT16:  printf("= %u\n", kv->value.uint16); break;
            case GGUF_TYPE_INT16:   printf("= %d\n", kv->value.int16); break;
            case GGUF_TYPE_UINT32:  printf("= %u\n", kv->value.uint32); break;
            case GGUF_TYPE_INT32:   printf("= %d\n", kv->value.int32); break;
            case GGUF_TYPE_UINT64:  printf("= %lu\n", kv->value.uint64); break;
            case GGUF_TYPE_INT64:   printf("= %ld\n", kv->value.int64); break;
            case GGUF_TYPE_FLOAT32: printf("= %f\n", kv->value.float32); break;
            case GGUF_TYPE_FLOAT64: printf("= %lf\n", kv->value.float64); break;
            case GGUF_TYPE_BOOL:    printf("= %s\n", kv->value.bool_ ? "true" : "false"); break;
            case GGUF_TYPE_STRING:  printf("= %s\n", kv->value.str.data ? kv->value.str.data : "(null)"); break;
            case GGUF_TYPE_ARRAY:
                printf("arr[%s:%zu]: ", 
                    kv->value.arr.type == GGUF_TYPE_STRING ? "str" : 
                    kv->value.arr.type == GGUF_TYPE_INT32 ? "i32" :
                    kv->value.arr.type == GGUF_TYPE_UINT32 ? "u32" :
                    kv->value.arr.type == GGUF_TYPE_FLOAT32 ? "f32" :
                    kv->value.arr.type == GGUF_TYPE_INT64 ? "i64" : "?",
                    kv->value.arr.n);
                for (size_t j = 0; j < kv->value.arr.n && j < 8; j++) {
                    if (kv->value.arr.type == GGUF_TYPE_STRING) {
                        printf("%s ", kv->value.arr.v[j].str.data ? kv->value.arr.v[j].str.data : "(null)");
                    } else if (kv->value.arr.type == GGUF_TYPE_INT32) {
                        printf("%d ", kv->value.arr.v[j].int32);
                    } else if (kv->value.arr.type == GGUF_TYPE_UINT32) {
                        printf("%u ", kv->value.arr.v[j].uint32);
                    } else if (kv->value.arr.type == GGUF_TYPE_FLOAT32) {
                        printf("%f ", kv->value.arr.v[j].float32);
                    }
                }
                printf("\n");
                break;
            default:
                printf("(unhandled type %d)\n", kv->type);
        }
    }
    
    gguf_close(ctx);
    return 0;
}
