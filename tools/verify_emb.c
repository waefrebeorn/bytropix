/**
 * verify_emb.c — Verify token embedding lookup against reference.
 * Loads token_embd.weight from GGUF, extracts token 9419 ('Hello'),
 * compares with /tmp/our_emb.bin (from infer_text DUMP_EMB).
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define D_MODEL 2048

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.gguf\n", argv[0]);
        return 1;
    }

    gguf_ctx *ctx = gguf_open(argv[1]);
    if (!ctx) { fprintf(stderr, "Failed to open GGUF\n"); return 1; }

    // Find token_embd.weight
    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t) { fprintf(stderr, "No token_embd.weight\n"); return 1; }

    printf("token_embd: n_dims=%d dims=[", t->n_dims);
    int64_t ne = 1;
    for (int i = 0; i < t->n_dims; i++) {
        if (i > 0) printf(", ");
        printf("%ld", t->dims[i]);
        ne *= t->dims[i];
    }
    printf("] ggml_type=%d ne=%ld\n", t->ggml_type, ne);

    int vocab_sz = ne / D_MODEL;
    printf("vocab_sz = %d\n", vocab_sz);

    // Allocate and read entire embedding table
    float *embd = (float *)malloc(ne * sizeof(float));
    if (!embd) { fprintf(stderr, "OOM\n"); return 1; }

    int n_read = gguf_read_tensor_f32(ctx, t, embd, ne);
    printf("Read %d floats\n", n_read);

    // Test tokens: BOS=248044, Hello=9419
    int test_tokens[] = {248044, 9419};
    const char *names[] = {"BOS", "Hello"};
    for (int ti = 0; ti < 2; ti++) {
        int id = test_tokens[ti];
        float *e = embd + id * D_MODEL;
        float mn=1e30, mx=-1e30, sum=0, sumsq=0;
        for (int i = 0; i < D_MODEL; i++) {
            if (e[i] < mn) mn = e[i];
            if (e[i] > mx) mx = e[i];
            sum += e[i];
            sumsq += e[i] * e[i];
        }
        printf("Token %d (%s): min=%.4f max=%.4f mean=%.4f rms=%.4f\n",
               id, names[ti], mn, mx, sum/D_MODEL, sqrtf(sumsq/D_MODEL));
        printf("  first 5: %.4f %.4f %.4f %.4f %.4f\n", e[0], e[1], e[2], e[3], e[4]);
    }

    // Compare against dumped embedding for token 9419
    float *ref_emb = (float *)malloc(D_MODEL * sizeof(float));
    FILE *f = fopen("/tmp/our_emb.bin", "rb");
    if (f) {
        fread(ref_emb, sizeof(float), D_MODEL, f);
        fclose(f);
        
        float *our_e = embd + 9419 * D_MODEL;
        float dot=0, n1=0, n2=0, maxdiff=0, meandiff=0;
        for (int i = 0; i < D_MODEL; i++) {
            dot += our_e[i] * ref_emb[i];
            n1 += our_e[i] * our_e[i];
            n2 += ref_emb[i] * ref_emb[i];
            float d = fabsf(our_e[i] - ref_emb[i]);
            if (d > maxdiff) maxdiff = d;
            meandiff += d;
        }
        meandiff /= D_MODEL;
        float cos = dot / (sqrtf(n1) * sqrtf(n2));
        printf("\n=== Embedding Comparison (token 9419) ===\n");
        printf("Our cos: %.6f\n", cos);
        printf("Our RMS: %.4f, Dump RMS: %.4f\n", sqrtf(n1/D_MODEL), sqrtf(n2/D_MODEL));
        printf("Max diff: %.6f, Mean diff: %.6f\n", maxdiff, meandiff);
        
        // Bits match perfectly?
        if (cos > 0.9999f) {
            printf("\n✅ EMBEDDING MATCH!\n");
        } else if (cos > 0.9f) {
            printf("\n⚠️ Embedding close but not exact (cos=%f)\n", cos);
        } else {
            printf("\n❌ Embedding DIVERGES (cos=%f)\n", cos);
        }
    } else {
        printf("\nNo /tmp/our_emb.bin found — run infer_text with DUMP_EMB first\n");
    }

    gguf_close(ctx);
    free(embd);
    free(ref_emb);
    return 0;
}
