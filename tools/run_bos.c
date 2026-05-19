/**
 * run_bos.c — Single BOS forward pass (no tokenizer needed).
 * Usage: DUMP_LAYER_DIR=/tmp/dump_layers_our ./run_bos [token_id]
 */
#include "wubu_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *model_path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int token_id = 248044;
    if (argc > 1) token_id = atoi(argv[1]);

    // Init model
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, model_path)) {
        fprintf(stderr, "Failed to init model\n");
        return 1;
    }
    mdl.enable_moe = true;

    int D = D_MODEL;
    int vs = mdl.vocab_size;

    // Read embedding for the single token (handles both file and in-memory)
    float *embd = (float *)malloc(D * sizeof(float));
    int got_emb = 0;
    if (mdl.use_embedding_file) {
        // Read from the embedding file
        const char *emb_path = "data/qwen36_embeddings_c.bin.raw";
        FILE *emb_f = fopen(emb_path, "rb");
        if (emb_f) {
            fseek(emb_f, (long long)token_id * D * sizeof(float), SEEK_SET);
            got_emb = (fread(embd, sizeof(float), D, emb_f) == (size_t)D);
            fclose(emb_f);
        }
    } else if (mdl.token_embd && token_id >= 0 && token_id < vs) {
        memcpy(embd, mdl.token_embd + (long long)token_id * D, D * sizeof(float));
        got_emb = 1;
    }

    if (!got_emb) {
        fprintf(stderr, "Failed to read embedding for token %d\n", token_id);
        free(embd);
        wubu_model_free(&mdl);
        return 1;
    }
    fprintf(stderr, "Token %d embedding read (%d floats)\n", token_id, D);

    // Forward pass: 1 batch, 1 token
    float *logits = (float *)malloc(vs * sizeof(float));
    wubu_model_forward_from_embd(&mdl, embd, 1, 1, logits);

    // Save logits if env var set (for cross-reference comparison)
    const char *logits_out = getenv("REF_LOGITS_PATH");
    if (logits_out) {
        FILE *fl = fopen(logits_out, "wb");
        if (fl) {
            fwrite(logits, sizeof(float), vs, fl);
            fclose(fl);
            fprintf(stderr, "Saved logits to %s\n", logits_out);
        }
    }

    fprintf(stderr, "Forward pass complete.\n");

    // Cleanup
    free(logits);
    free(embd);
    wubu_model_free(&mdl);
    return 0;
}
