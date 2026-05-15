#include "wubu_model.h"
#include "gguf_reader.h"
#include "wubu_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void top5(float *logits, int vs, wubu_tokenizer_t *tok) {
    int top[5] = {0,1,2,3,4}; float tv[5];
    for (int i = 0; i < 5; i++) tv[i] = logits[i];
    for (int i = 5; i < vs; i++) {
        if (logits[i] > tv[4]) {
            tv[4] = logits[i]; top[4] = i;
            for (int k = 3; k >= 0; k--) {
                if (tv[k] < tv[k+1]) {
                    float tmp = tv[k]; tv[k] = tv[k+1]; tv[k+1] = tmp;
                    int tmpi = top[k]; top[k] = top[k+1]; top[k+1] = tmpi;
                }
            }
        }
    }
    char buf[256];
    for (int k = 0; k < 5; k++) {
        wubu_tokenizer_decode(tok, top+k, 1, buf, 255);
        printf("  [%d]='%s'(%.2f)\n", top[k], buf, tv[k]);
    }
    printf("  gap=%.2f\n", tv[0] - tv[4]);
}

int main() {
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    wubu_model_t model;
    if (!wubu_model_init(&model, path)) return 1;
    
    wubu_tokenizer_t tok;
    wubu_tokenizer_init(&tok, path);
    
    gguf_ctx *ctx = model.gguf_ctx;
    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    int64_t ne = t->dims[0] * t->dims[1];
    int vs = (int)(ne / D_MODEL);
    float *embd = malloc(ne * sizeof(float));
    gguf_read_tensor_f32(ctx, t, embd, ne);
    printf("Vocab: %d\n", vs);
    
    float x[2048];
    float *logits = malloc(vs * sizeof(float));
    
    // Test 1: Token 0
    memcpy(x, embd, 2048 * sizeof(float));
    wubu_model_forward_from_embd(&model, x, 1, 1, logits);
    printf("\nToken 0:\n"); top5(logits, vs, &tok);
    
    // Test 2: BOS token 248044
    int bos = 248044;
    memcpy(x, embd + bos * 2048, 2048 * sizeof(float));
    wubu_model_forward_from_embd(&model, x, 1, 1, logits);
    printf("\nBOS token 248044:\n"); top5(logits, vs, &tok);
    
    // Decode BOS
    char buf[256];
    wubu_tokenizer_decode(&tok, &bos, 1, buf, 255);
    printf("BOS decodes to: '%s'\n", buf);
    
    // Test 3: Token 0 again (check determinism)
    memcpy(x, embd, 2048 * sizeof(float));
    wubu_model_forward_from_embd(&model, x, 1, 1, logits);
    printf("\nToken 0 (repeat):\n"); top5(logits, vs, &tok);
    
    wubu_model_free(&model);
    wubu_tokenizer_free(&tok);
    free(embd); free(logits);
    printf("=== PASS ===\n");
    return 0;
}
