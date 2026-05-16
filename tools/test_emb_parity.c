/**
 * test_emb_parity.c — Compare token embedding from our gguf_reader vs llama.cpp.
 * Build: gcc -O2 -Iinclude -I/home/wubu/llama.cpp/include -I/home/wubu/llama.cpp/ggml/include \
 *        -o test_emb_parity tools/test_emb_parity.c src/gguf_reader.o \
 *        -L/home/wubu/llama.cpp/build/bin -Wl,-rpath,/home/wubu/llama.cpp/build/bin \
 *        -lllama -lggml-base -lggml-cpu -lggml-cuda -lm -lstdc++ -lssl -lcrypto
 */
#include "gguf_reader.h"
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s model.gguf token_id\n", argv[0]); return 1; }
    int target_id = atoi(argv[2]);
    const char *path = argv[1];

    // === Our gguf_reader ===
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open GGUF\n"); return 1; }
    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t) { fprintf(stderr, "No token_embd.weight\n"); return 1; }
    int64_t n_elems = 1;
    for (int d = 0; d < t->n_dims; d++) n_elems *= t->dims[d];
    printf("Our tensor: n_dims=%d dims=[", t->n_dims);
    for (int d = 0; d < t->n_dims; d++) printf("%lld%s", t->dims[d], d+1<t->n_dims?",":"]");
    printf(" type=%d\n", t->ggml_type);
    
    float *our_emb = (float *)malloc(n_elems * sizeof(float));
    int n_read = gguf_read_tensor_f32(ctx, t, our_emb, n_elems);
    if (n_read <= 0) { fprintf(stderr, "Failed to read tensor\n"); return 1; }
    gguf_close(ctx);
    
    int vs = (int)(n_elems / 2048);
    if (target_id < 0 || target_id >= vs) { fprintf(stderr, "Token ID out of range\n"); return 1; }
    float *our_hello = our_emb + target_id * 2048LL;
    printf("Our emb[%d]: mean=%.6f max=%.6f min=%.6f\n", target_id,
           our_hello[0], our_hello[1], our_hello[2]); // quick check

    // === llama.cpp ===
    struct llama_model_params mp = llama_model_default_params();
    struct llama_model *lm = llama_model_load_from_file(path, mp);
    if (!lm) { fprintf(stderr, "Failed to load model with llama\n"); return 1; }
    
    // Get embedding directly from model
    // llama_model doesn't expose token_embd directly, so we need a workaround
    // Use tokenize + context to get embedding
    struct llama_context_params cp = llama_context_default_params();
    cp.embeddings = true;
    cp.n_ctx = 128;
    struct llama_context *lctx = llama_init_from_model(lm, cp);
    if (!lctx) { fprintf(stderr, "Failed to create context\n"); return 1; }
    
    // Tokenize a single token (just the target token ID)
    llama_token tokens[2] = {target_id};
    llama_batch batch = llama_batch_get_one(tokens, 1);
    if (llama_decode(lctx, batch) != 0) { fprintf(stderr, "llama_decode failed\n"); return 1; }
    
    // Get embeddings - this gives hidden state AFTER all layers
    float *ref_emb = llama_get_embeddings(lctx);
    if (!ref_emb) { fprintf(stderr, "No embeddings\n"); return 1; }
    printf("llama emb[%d] (after all layers): mean=%.6f max=%.6f min=%.6f\n", target_id,
           ref_emb[0], ref_emb[1], ref_emb[2]);

    // Print first 10 values of both
    printf("\nFirst 10 our Hello emb: ");
    for (int i = 0; i < 10; i++) printf("%.6f ", our_hello[i]);
    printf("\nFirst 10 llama after 0 layers (1 tok): ");
    for (int i = 0; i < 10; i++) printf("%.6f ", ref_emb[i]);
    printf("\n");

    // Compare cos sim of the two hidden states  
    double dot = 0, n1 = 0, n2 = 0;
    for (int i = 0; i < 2048; i++) {
        dot += our_hello[i] * ref_emb[i];
        n1 += our_hello[i] * our_hello[i];
        n2 += ref_emb[i] * ref_emb[i];
    }
    printf("\nToken %d: our_initial_emb vs llama_final_hidden cos_sim = %.4f\n",
           target_id, dot / sqrt(n1*n2));
    
    // Also dump the initial embedding from llama - we need to get the actual weight table
    // Use the model's internal tensor
    printf("\nNOTE: llama_get_embeddings gives FINAL hidden (after 40 layers), not initial embedding.\n");
    printf("To compare initial embeddings, we'd need raw weight access.\n");

    llama_free(lctx);
    llama_model_free(lm);
    free(our_emb);
    return 0;
}
