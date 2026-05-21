// ref_dumper_mtp.cpp — MTP head output cross-reference
// Links against libllama.so to get reference MTP logits for comparison.
// Usage: DUMP_LAYER_DIR=/tmp/dump_layers ./ref_dumper_mtp model.gguf [token_id]
#include "llama.h"
#include "llama-ext.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.gguf [token_id]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    int token_id = 248044;
    if (argc >= 3) token_id = atoi(argv[2]);

    ggml_backend_load_all();

    auto mparams = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    int n_embd = llama_model_n_embd(model);

    // === TARGET CONTEXT (main model) ===
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 1;
    cparams.n_ubatch = 1;

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to init context\n"); return 1; }

    int n_threads = 16;
    const char * nt_env = getenv("LLAMA_N_THREADS");
    if (nt_env) n_threads = atoi(nt_env);
    llama_set_n_threads(ctx, n_threads, n_threads);

    // Enable pre-norm embeddings
    llama_set_embeddings(ctx, true);
    llama_set_embeddings_pre_norm(ctx, true, false);

    // Run target forward
    llama_token tokens[] = { (llama_token)token_id };
    auto batch = llama_batch_get_one(tokens, 1);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Target decode failed\n");
        llama_model_free(model);
        return 1;
    }

    // Get target logits and pre-norm embeddings
    float * target_logits = llama_get_logits(ctx);
    float * h_pre_norm = llama_get_embeddings_pre_norm(ctx);

    fprintf(stderr, "Target forward done. n_embd=%d n_vocab=%d\n", n_embd, n_vocab);

    // Save target logits
    FILE * f = fopen("/tmp/ref_target_logits.bin", "wb");
    if (f) { fwrite(target_logits, sizeof(float), n_vocab, f); fclose(f); }

    // Save pre-norm embedding
    f = fopen("/tmp/ref_h_pre_norm.bin", "wb");
    if (f) { fwrite(h_pre_norm, sizeof(float), n_embd, f); fclose(f); }
    fprintf(stderr, "Saved target logits and h_pre_norm\n");

    // Print target argmax
    int target_argmax = 0; float tmax = target_logits[0];
    for (int i = 1; i < n_vocab; i++) if (target_logits[i] > tmax) { tmax = target_logits[i]; target_argmax = i; }
    fprintf(stderr, "Target argmax: %d (val=%.2f)\n", target_argmax, tmax);

    // === MTP CONTEXT ===
    auto cparams_mtp = llama_context_default_params();
    cparams_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cparams_mtp.n_ctx = 512;
    cparams_mtp.n_batch = 1;
    cparams_mtp.n_ubatch = 1;
    cparams_mtp.n_rs_seq = 0;

    struct llama_context * ctx_mtp = llama_init_from_model(model, cparams_mtp);
    if (!ctx_mtp) { fprintf(stderr, "Failed to init MTP context\n"); llama_model_free(model); return 1; }
    llama_set_n_threads(ctx_mtp, n_threads, n_threads);
    llama_set_embeddings_pre_norm(ctx_mtp, true, true);

    // Create batch with embd = h_pre_norm, token = token_id
    auto batch_mtp = llama_batch_init(1, 0, 1);
    batch_mtp.n_tokens = 1;
    batch_mtp.token[0] = (llama_token)token_id;
    batch_mtp.embd = (float *)malloc(n_embd * sizeof(float));
    memcpy(batch_mtp.embd, h_pre_norm, n_embd * sizeof(float));
    batch_mtp.pos[0] = 0;
    batch_mtp.n_seq_id[0] = 1;
    batch_mtp.seq_id[0][0] = 0;
    batch_mtp.logits[0] = 1;

    if (llama_decode(ctx_mtp, batch_mtp) != 0) {
        fprintf(stderr, "MTP decode failed\n");
        llama_batch_free(batch_mtp);
        llama_model_free(model);
        return 1;
    }

    // Get MTP logits
    float * mtp_logits = llama_get_logits_ith(ctx_mtp, -1);

    // Save MTP logits
    f = fopen("/tmp/ref_mtp_logits.bin", "wb");
    if (f) { fwrite(mtp_logits, sizeof(float), n_vocab, f); fclose(f); }
    fprintf(stderr, "Saved MTP logits\n");

    // Print MTP argmax
    int mtp_argmax = 0; float mmax = mtp_logits[0];
    for (int i = 1; i < n_vocab; i++) if (mtp_logits[i] > mmax) { mmax = mtp_logits[i]; mtp_argmax = i; }
    fprintf(stderr, "MTP argmax: %d (val=%.2f)\n", mtp_argmax, mmax);

    // Compare
    if (target_argmax == mtp_argmax) {
        fprintf(stderr, "*** MATCH! Both predict token %d ***\n", target_argmax);
    } else {
        fprintf(stderr, "*** MISMATCH! Target=%d MTP=%d ***\n", target_argmax, mtp_argmax);
        // Check if target's token is in MTP's top-K
        float target_val = mtp_logits[target_argmax];
        int rank = 1;
        for (int i = 0; i < n_vocab; i++) if (mtp_logits[i] > target_val) rank++;
        fprintf(stderr, "Target token %d rank in MTP: %d/%d\n", target_argmax, rank, n_vocab);
    }

    // Also get MTP pre-norm for comparison
    float * mtp_h = llama_get_embeddings_pre_norm_ith(ctx_mtp, 0);
    if (mtp_h) {
        double h_diff = 0;
        for (int i = 0; i < n_embd; i++) h_diff += fabs(h_pre_norm[i] - mtp_h[i]);
        fprintf(stderr, "h_pre_norm vs MTP h: avg diff=%.6f\n", h_diff / n_embd);
        f = fopen("/tmp/ref_mtp_hidden.bin", "wb");
        if (f) { fwrite(mtp_h, sizeof(float), n_embd, f); fclose(f); }
    }

    llama_batch_free(batch_mtp);
    llama_model_free(model);
    return 0;
}
