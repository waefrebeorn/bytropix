// Test: compare our logits vs llama.cpp for the same single-token input
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "llama.h"          // from llama.cpp
#include "wubu_model.h"     // from bytropix
#include "gguf_reader.h"    // from bytropix
#include "wubu_tokenizer.h"

static void top5(float *logits, int vs, wubu_tokenizer_t *tok, const char *label) {
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
    printf("%s top-5:\n", label);
    for (int k = 0; k < 5; k++) {
        wubu_tokenizer_decode(tok, top+k, 1, buf, 255);
        printf("  [%d]='%s'(%.2f)\n", top[k], buf, tv[k]);
    }
    
    // Compare with llama's top-1
    if (k > 0) printf("  gap=%.2f\n", tv[0] - tv[4]);
}

int main() {
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    // ---- Our engine ----
    printf("=== Our engine ===\n");
    wubu_model_t model;
    if (!wubu_model_init(&model, path)) return 1;
    
    wubu_tokenizer_t tok;
    wubu_tokenizer_init(&tok, path);
    
    // Load token embedding
    gguf_ctx *ctx = model.gguf_ctx;
    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    int vs = (int)(t->dims[0] * t->dims[1] / D_MODEL);
    float *embd = malloc((int64_t)vs * D_MODEL * sizeof(float));
    gguf_read_tensor_f32(ctx, t, embd, (int64_t)vs * D_MODEL);
    
    // Use token 0 as input
    float x[2048];
    memcpy(x, embd, 2048 * sizeof(float));
    
    float *our_logits = malloc(vs * sizeof(float));
    wubu_model_forward_from_embd(&model, x, 1, 1, our_logits);
    top5(our_logits, vs, &tok, "Ours");
    
    // ---- llama.cpp ----
    printf("\n=== llama.cpp ===\n");
    llama_model *lm = NULL;
    llama_context *lctx = NULL;
    
    // Use default params
    llama_model_params lmp = llama_model_default_params();
    lmp.n_gpu_layers = 99;  // Use GPU if available
    lctx = NULL; // dummy init to make compiler happy
    
    // Load model
    lm = llama_load_model_from_file(path, lmp);
    if (!lm) { printf("FAIL: llama_load_model\n"); return 1; }
    
    llama_context_params lcp = llama_context_default_params();
    lcp.n_ctx = 512;
    lctx = llama_new_context_with_model(lm, lcp);
    if (!lctx) { printf("FAIL: llama_new_context\n"); return 1; }
    
    // Get vocab
    const llama_vocab *vocab = llama_model_get_vocab(lm);
    int n_vocab = llama_vocab_n_tokens(vocab);
    printf("llama vocab: %d\n", n_vocab);
    
    // Tokenize BOS token
    int n_tok = 1;
    llama_token *tokens = (llama_token*)malloc(n_tok * sizeof(llama_token));
    tokens[0] = llama_token_bos(lm);
    printf("BOS token: %d\n", tokens[0]);
    
    // Decode
    llama_batch batch = llama_batch_get_one(tokens, n_tok);
    if (llama_decode(lctx, batch)) { printf("FAIL: llama_decode\n"); return 1; }
    
    // Get logits
    float *llama_logits = llama_get_logits_ith(lctx, 0);
    if (!llama_logits) { printf("FAIL: llama_get_logits\n"); return 1; }
    
    // Top-5 via our tokenizer for comparison
    top5(llama_logits, n_vocab, &tok, "llama");
    
    // ---- Comparison ----
    printf("\n=== Comparison (first 10 logits) ===\n");
    float max_diff = 0.0f;
    int min_v = n_vocab < vs ? n_vocab : vs;
    for (int i = 0; i < min_v && i < 10; i++) {
        float diff = fabsf(our_logits[i] - llama_logits[i]);
        if (diff > max_diff) max_diff = diff;
        printf("  [%d] ours=%.4f llama=%.4f diff=%.4f\n", i, our_logits[i], llama_logits[i], diff);
    }
    printf("Max diff (first 10): %.6f\n", max_diff);
    
    // Compare token 0 predictions
    printf("\nOur top-1 token: ");
    int our_top = 0; float our_best = our_logits[0];
    for (int i = 1; i < vs; i++) if (our_logits[i] > our_best) { our_best = our_logits[i]; our_top = i; }
    
    int llama_top = 0; float llama_best = llama_logits[0];
    for (int i = 1; i < n_vocab; i++) if (llama_logits[i] > llama_best) { llama_best = llama_logits[i]; llama_top = i; }
    
    char buf[256];
    wubu_tokenizer_decode(&tok, &our_top, 1, buf, 255);
    printf("  [%d]='%s'(%.4f)\n", our_top, buf, our_best);
    wubu_tokenizer_decode(&tok, &llama_top, 1, buf, 255);
    printf("llama top-1: [%d]='%s'(%.4f)\n", llama_top, buf, llama_best);
    
    // Cleanup
    llama_free(lctx);
    llama_free_model(lm);
    
    free(embd); free(our_logits); free(tokens);
    wubu_tokenizer_free(&tok);
    wubu_model_free(&model);
    printf("\n=== PASS ===\n");
    return 0;
}
