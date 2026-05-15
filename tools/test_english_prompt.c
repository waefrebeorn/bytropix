#include "wubu_model.h"
#include "gguf_reader.h"
#include "wubu_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void top5(float *logits, int vs, wubu_tokenizer_t *tok) {
    int top[5] = {0}; float tv[5] = {-1e30,-1e30,-1e30,-1e30,-1e30};
    for (int j = 0; j < vs; j++) {
        if (logits[j] > tv[4]) {
            tv[4] = logits[j]; top[4] = j;
            for (int k = 3; k >= 0; k--) {
                if (tv[k] < tv[k+1]) {
                    float tmp = tv[k]; tv[k] = tv[k+1]; tv[k+1] = tmp;
                    int ti = top[k]; top[k] = top[k+1]; top[k+1] = ti;
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
    int vs = (int)(t->dims[0] * t->dims[1] / D_MODEL);
    float *embd = malloc((int64_t)vs * D_MODEL * sizeof(float));
    gguf_read_tensor_f32(ctx, t, embd, (int64_t)vs * D_MODEL);
    
    const char *prompts[] = {"Hello! How are you?", "The capital of France is", "Once upon a time", "2 + 2 =", "Translate to French: Hello"};
    
    for (int pi = 0; pi < 5; pi++) {
        const char *prompt = prompts[pi];
        int pids[2048];
        int np = wubu_tokenizer_encode(&tok, prompt, pids, 2048);
        if (np <= 0) { printf("Skipped %s\n", prompt); continue; }
        
        float *x = (float *)malloc(np * D_MODEL * sizeof(float));
        for (int i = 0; i < np; i++) {
            int id = pids[i];
            memcpy(x + i * D_MODEL, embd + (id >= 0 && id < vs ? id : 0) * D_MODEL, D_MODEL * sizeof(float));
        }
        
        float *logits = (float *)malloc(np * vs * sizeof(float));
        wubu_model_forward_from_embd(&model, x, 1, np, logits);
        
        char ps[50]; strncpy(ps, prompt, 45); ps[45] = 0;
        printf("Prompt: \"%s\" (%d tok)\n", ps, np);
        top5(logits + (np-1) * vs, vs, &tok);
        printf("\n");
        
        free(x); free(logits);
    }
    
    free(embd);
    wubu_tokenizer_free(&tok);
    wubu_model_free(&model);
    printf("=== PASS ===\n");
    return 0;
}
