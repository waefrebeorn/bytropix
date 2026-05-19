/**
 * gen_text_mtp.c — MTP speculative decode with verification.
 * DRAFT_N=2 (blog: 83% acceptance at 2 drafts, drops to 50% at 4).
 *
 * LOADS TWO MODEL FILES:
 *   Main: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (regular model, layers 0-39)
 *   MTP:  /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf (blk.40 head only)
 *
 * These are DIFFERENT quantizations — base weights are only in the regular file.
 *
 * Flow per iteration:
 *   1. Emit main model's prediction (argmax of last_logits)
 *   2. Generate 2 draft tokens via MTP head (blk.40 from MTP model)
 *   3. Checkpoint model state
 *   4. Forward main_token through main → verify_logits
 *   5. Check draft[1] against main's prediction:
 *      - MATCH: emit draft[1] (2 tokens for 1 main forward) ✓
 *      - MISMATCH: rollback, emit main's real prediction instead
 *   6. Update h_39 + last_logits for next iteration
 *
 * Build: make gen_text_mtp
 * Usage: MTP=1 OMP_NUM_THREADS=16 ./gen_text_mtp "Hello" 64
 * Without MTP: behaves like gen_text (1 token per step)
 * With MTP: 2-draft speculative decode with accept/reject
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "wubu_tokenizer.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>

#define DRAFT_N 2  // Blog: 83% acceptance at 2, 50% at 4. Max 2 recommended.

static volatile int g_stop = 0;
static void handle_sigint(int sig) { (void)sig; g_stop = 1; }
static double wall_clock(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int argmax(float *logits, int vs) {
    int mi = 0; float mv = logits[0];
    for (int i = 1; i < vs; i++) if (logits[i] > mv) { mv = logits[i]; mi = i; }
    return mi;
}

static void get_embd(wubu_model_t *mdl, int token, float *out, FILE *emb_file) {
    int D = D_MODEL;
    if (mdl->use_embedding_file && emb_file) {
        fseek(emb_file, (long long)token * D * sizeof(float), SEEK_SET);
        size_t n = fread(out, sizeof(float), D, emb_file);
        if (n != (size_t)D) memset(out, 0, D * sizeof(float));
    } else if (mdl->token_embd) {
        memcpy(out, mdl->token_embd + (long long)token * D, D * sizeof(float));
    } else {
        memset(out, 0, D * sizeof(float));
    }
}

int main(int argc, char **argv) {
    const char *main_model = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *mtp_model  = "/models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf";
    const char *prompt = "The meaning of life is";
    int max_tokens = 32;
    int D = D_MODEL;
    int use_mtp = getenv("MTP") != NULL;
    int verbose = getenv("VERBOSE") != NULL;

    if (argc > 1) prompt = argv[1];
    if (argc > 2) max_tokens = atoi(argv[2]);

    signal(SIGINT, handle_sigint);

    // ====== Load MAIN model (regular GGUF, layers 0-39) ======
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, main_model)) return 1;
    mdl.enable_moe = true;

    // ====== Load MTP head from SEPARATE MTP model file ======
    gguf_ctx *mtp_ctx = NULL;
    const uint8_t *mtp_blob = NULL;
    if (use_mtp) {
        mtp_ctx = gguf_open(mtp_model);
        if (!mtp_ctx) {
            fprintf(stderr, "Failed to open MTP model: %s\n", mtp_model);
            use_mtp = 0;
        } else {
            fprintf(stderr, "Opened MTP model: %s\n", mtp_model);
            gguf_buffer_data(mtp_ctx);
            mtp_blob = (const uint8_t *)mtp_ctx->data_blob;

            if (!wubu_mtp_load(&mdl.mtp, mtp_model, mtp_ctx, mtp_blob)) {
                fprintf(stderr, "MTP head not available in MTP model file\n");
                gguf_close(mtp_ctx);
                mtp_ctx = NULL;
                use_mtp = 0;
            } else {
                fprintf(stderr, "MTP head loaded (%d draft tokens)\n", DRAFT_N);
            }
        }
    }

    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, main_model)) { wubu_model_free(&mdl); return 1; }

    int vs = mdl.vocab_size;
    int prompt_tokens[1024];
    int n_prompt = wubu_tokenizer_encode(&tok, prompt, prompt_tokens, 1024);
    if (n_prompt <= 0) { prompt_tokens[0] = tok.bos_id >= 0 ? tok.bos_id : 248044; n_prompt = 1; }
    fprintf(stderr, "Prompt: %d tokens\n", n_prompt);

    // Embeddings for prompt
    float *embd = (float *)malloc(n_prompt * D * sizeof(float));
    FILE *emb_file = NULL;
    if (mdl.use_embedding_file) {
        emb_file = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (!emb_file) { free(embd); wubu_model_free(&mdl); return 1; }
    }
    for (int i = 0; i < n_prompt; i++)
        get_embd(&mdl, prompt_tokens[i], embd + i * D, emb_file);

    // Prefill: get logits + h_39 via save_last_hidden
    float *logits_buf = (float *)malloc(n_prompt * vs * sizeof(float));
    float h_39[D];
    double t0 = wall_clock();
    mdl.save_last_hidden = h_39;
    wubu_model_forward_from_embd(&mdl, embd, 1, n_prompt, logits_buf);
    mdl.save_last_hidden = NULL;
    double t_prefill = wall_clock() - t0;
    fprintf(stderr, "Prefill: %.2fs (%.1f tok/s)\n", t_prefill, n_prompt / t_prefill);

    // Print prompt
    { char buf[2048]; int nc = wubu_tokenizer_decode(&tok, prompt_tokens, n_prompt, buf, 2048);
      if (nc > 0) fprintf(stderr, "Input: %s\n", buf); }

    float *logits_out = (float *)malloc(vs * sizeof(float));
    float *cur = (float *)malloc(D * sizeof(float));
    float *mtp_logits = (float *)malloc(vs * sizeof(float));
    float *logit_correction = (float *)calloc(vs, sizeof(float));  // EMA correction for MTP logits
    float *prev_cur = (float *)malloc(D * sizeof(float));
    int total_gen = 0;
    int total_accepted = 0;
    int total_attempted = 0;

    // After prefill: last_logits predicts first generated token
    // prev_cur = embedding of the LAST PROMPT token (for MTP's first draft)
    float *last_logits = logits_buf + (n_prompt - 1) * vs;
    get_embd(&mdl, prompt_tokens[n_prompt - 1], prev_cur, emb_file);
    memcpy(cur, prev_cur, D * sizeof(float));

    // ============================================================
    // Main decode loop
    // ============================================================
    while (total_gen < max_tokens && !g_stop) {
        // ====== Emit main model's prediction ======
        int main_token = argmax(last_logits, vs);
        {
            char piece[256];
            int nc = wubu_tokenizer_decode(&tok, &main_token, 1, piece, 256);
            if (nc > 0) fwrite(piece, 1, nc, stdout); else printf("<%d>", main_token);
            fflush(stdout);
        }
        total_gen++;
        if (main_token == tok.eos_id || main_token == tok.bos_id) break;
        if (total_gen >= max_tokens || g_stop) break;

        // ====== Attempt speculative decode ======
        int accepted_drafts = 0;
        (void)accepted_drafts; // suppress unused warning (used inside MTP block)
        if (use_mtp && mdl.mtp.loaded) {
            total_attempted++;

            // Reset MTP KV cache for fresh draft generation
            mdl.mtp.cache_len = 0;

            // Generate draft[0] using prev_cur (token before the one we just predicted)
            // This should predict main_token (same position as last_logits)
            wubu_mtp_draft_forward(&mdl, h_39, prev_cur, 1, mtp_logits);
            
            // Save raw MTP logits before correction (for EMA update)
            float mtp_raw[256];  // Only first 256 logits needed for EMA (practical approximation)
            int mtp_raw_n = (vs < 256) ? vs : 256;
            for (int v = 0; v < mtp_raw_n; v++) mtp_raw[v] = mtp_logits[v];
            
            // Apply online logit correction to compensate for quantization bias
            for (int v = 0; v < vs; v++) mtp_logits[v] += logit_correction[v];
            
            int draft0 = argmax(mtp_logits, vs);

            if (draft0 == main_token) {
                // draft[0] matches main model! Log this and update correction EMA.
                // Update EMA correction: correction = 0.9*c + 0.1*(main_logits - mtp_raw_logits)
                // We use corrected mtp_logits here (slower adaptation but simpler code)
                if (mtp_raw_n > 0) {
                    for (int v = 0; v < mtp_raw_n; v++) {
                        float diff = last_logits[v] - mtp_raw[v];
                        logit_correction[v] = 0.9f * logit_correction[v] + 0.1f * diff;
                    }
                }
                // draft[0] matches main model! Generate draft[1]
                float mid_cur[D];
                get_embd(&mdl, main_token, mid_cur, emb_file);

                // Generate draft[1] via MTP head
                wubu_mtp_draft_forward(&mdl, h_39, mid_cur, 1, mtp_logits);
                
                // Save raw MTP logits before correction (for EMA update)
                float mtp_raw1[256];
                int mtp_raw1_n = (vs < 256) ? vs : 256;
                for (int v = 0; v < mtp_raw1_n; v++) mtp_raw1[v] = mtp_logits[v];
                
                // Apply logit correction before sampling
                for (int v = 0; v < vs; v++) mtp_logits[v] += logit_correction[v];
                
                int draft1 = argmax(mtp_logits, vs);

                // Checkpoint current model state before verifying draft[1]
                if (!wubu_model_checkpoint(&mdl)) {
                    if (verbose) fprintf(stderr, "\n[MTP] Checkpoint failed\n");
                    goto normal_advance;
                }

                // Forward main_token through main model to get its prediction
                // This advances the model's state past main_token
                mdl.save_last_hidden = h_39;
                wubu_model_forward_from_embd(&mdl, mid_cur, 1, 1, logits_out);
                mdl.save_last_hidden = NULL;
                int main_next = argmax(logits_out, vs);
                
                // Update EMA correction from draft[1] verification
                if (mtp_raw1_n > 0) {
                    for (int v = 0; v < mtp_raw1_n; v++) {
                        float diff = logits_out[v] - mtp_raw1[v];
                        logit_correction[v] = 0.9f * logit_correction[v] + 0.1f * diff;
                    }
                }

                if (main_next == draft1) {
                    // BOTH drafts accepted! We emitted main_token already, now emit draft[1]
                    total_accepted++;
                    {
                        char piece[256];
                        int nc = wubu_tokenizer_decode(&tok, &draft1, 1, piece, 256);
                        if (nc > 0) fwrite(piece, 1, nc, stdout); else printf("<%d>", draft1);
                        fflush(stdout);
                    }
                    total_gen++;
                    if (verbose) fprintf(stderr, "\n[MTP] 2/2 accepted ✓\n");

                    // h_39 now reflects position after main_token
                    // We need to advance one more step for next iteration
                    get_embd(&mdl, draft1, cur, emb_file);
                    memcpy(prev_cur, mid_cur, D * sizeof(float));

                    if (total_gen < max_tokens && !g_stop &&
                        draft1 != tok.eos_id && draft1 != tok.bos_id) {
                        mdl.save_last_hidden = h_39;
                        wubu_model_forward_from_embd(&mdl, cur, 1, 1, logits_out);
                        mdl.save_last_hidden = NULL;
                    }
                    last_logits = logits_out;
                } else {
                    // draft[1] REJECTED — rollback to pre-main_token state
                    wubu_model_rollback(&mdl);
                    // We already emitted main_token. Now forward it and emit main's real next token.
                    mdl.save_last_hidden = h_39;
                    wubu_model_forward_from_embd(&mdl, mid_cur, 1, 1, logits_out);
                    mdl.save_last_hidden = NULL;

                    int real_next = argmax(logits_out, vs);
                    {
                        char piece[256];
                        int nc = wubu_tokenizer_decode(&tok, &real_next, 1, piece, 256);
                        if (nc > 0) fwrite(piece, 1, nc, stdout); else printf("<%d>", real_next);
                        fflush(stdout);
                    }
                    total_gen++;
                    if (verbose) fprintf(stderr, "\n[MTP] 1/2 accepted (draft[1] rejected: main=%d draft=%d)\n",
                                         main_next, draft1);

                    get_embd(&mdl, real_next, cur, emb_file);
                    memcpy(prev_cur, mid_cur, D * sizeof(float));
                    last_logits = logits_out;
                }
            } else {
                // draft[0] doesn't match main model — MTP out of sync
                // Still update EMA correction from available logits
                if (mtp_raw_n > 0) {
                    for (int v = 0; v < mtp_raw_n; v++) {
                        float diff = last_logits[v] - mtp_raw[v];
                        logit_correction[v] = 0.9f * logit_correction[v] + 0.1f * diff;
                    }
                }
                if (verbose) fprintf(stderr, "\\n[MTP] draft[0] mismatch: main=%d draft=%d\\n", main_token, draft0);
                goto normal_advance;
            }
        } else {
            normal_advance:
            // No MTP or fallback: normal single-token advance
            get_embd(&mdl, main_token, cur, emb_file);
            memcpy(prev_cur, cur, D * sizeof(float));

            mdl.save_last_hidden = h_39;
            wubu_model_forward_from_embd(&mdl, cur, 1, 1, logits_out);
            mdl.save_last_hidden = NULL;
            last_logits = logits_out;
        }

        if (main_token == tok.eos_id || main_token == tok.bos_id) break;
    }

    printf("\n");

    double t_total = wall_clock() - t0;
    fprintf(stderr, "\n--- Stats ---\n");
    fprintf(stderr, "Total: %d tok in %.2fs (%.1f tok/s)\n", total_gen + n_prompt, t_total,
           (total_gen + n_prompt) / t_total);
    fprintf(stderr, "Decode: %d tok in %.2fs (%.1f tok/s)\n", total_gen, t_total - t_prefill,
           total_gen / (t_total - t_prefill));
    if (total_attempted > 0) {
        fprintf(stderr, "MTP: %d attempts, %d accepted (%.0f%%), draft=%d\n",
               total_attempted, total_accepted,
               100.0 * total_accepted / total_attempted, DRAFT_N);
    }

    free(embd); free(logits_buf); free(logits_out); free(cur); free(mtp_logits); free(logit_correction); free(prev_cur);
    if (emb_file) fclose(emb_file);
    if (mtp_ctx) gguf_close(mtp_ctx);
    wubu_tokenizer_free(&tok);
    wubu_model_free(&mdl);
    return 0;
}
