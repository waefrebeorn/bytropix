/**
 * gen_text_mtp.c — MTP multi-token prediction inference.
 * MTP=1: free-tokens mode (main prediction + DRAFT_N MTP extra tokens per step).
 * Default: non-MTP (1 token per decode step, same as gen_text).
 *
 * Flow per iteration:
 *   1. Emit main model's prediction (argmax(last_logits) from prev forward)
 *   2. Generate MTP extra tokens via blk.40 head (updates cur each step)
 *   3. Forward last emitted token through main model → new h_39 + logits
 *
 * The MTP head predicts: given h_39 (from position P) + token at P, what's next?
 * First MTP draft = first generated token = should match main model's prediction.
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

#define DRAFT_N 4

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
    const char *model_path = "/models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf";
    const char *prompt = "The meaning of life is";
    int max_tokens = 32;
    int D = D_MODEL;
    int use_mtp = getenv("MTP") != NULL;

    if (argc > 1) prompt = argv[1];
    if (argc > 2) max_tokens = atoi(argv[2]);

    signal(SIGINT, handle_sigint);

    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, model_path)) return 1;
    mdl.enable_moe = true;

    if (use_mtp) {
        if (!wubu_mtp_load(&mdl.mtp, model_path, mdl.gguf_ctx, (const uint8_t*)mdl.gguf_ctx->data_blob)) {
            fprintf(stderr, "MTP head not available\n");
            use_mtp = 0;
        } else {
            fprintf(stderr, "MTP head loaded (opt-in)\n");
        }
    }

    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, model_path)) { wubu_model_free(&mdl); return 1; }

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
    fprintf(stderr, "Prefill: %.2fs\n", t_prefill);

    // Print prompt
    { char buf[2048]; int nc = wubu_tokenizer_decode(&tok, prompt_tokens, n_prompt, buf, 2048);
      if (nc > 0) fprintf(stderr, "Input: %s\n", buf); }

    float *logits_out = (float *)malloc(vs * sizeof(float));
    float *h_next = (float *)malloc(D * sizeof(float));
    float *cur = (float *)malloc(D * sizeof(float));
    float *mtp_logits = (float *)malloc(vs * sizeof(float));
    int total_gen = 0;

    // After prefill: last_logits predicts first generated token
    // cur = embedding of the LAST PROMPT token (for MTP's first draft)
    float *last_logits = logits_buf + (n_prompt - 1) * vs;
    get_embd(&mdl, prompt_tokens[n_prompt - 1], cur, emb_file);

    while (total_gen < max_tokens && !g_stop) {
        // ====== STEP 1: Emit main model's prediction ======
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

        // Update cur to this emitted token's embedding for MTP
        get_embd(&mdl, main_token, cur, emb_file);

        // VERIFY: MTP head should predict SAME first token as main model
        // when given h_39 + embd(last_prompt_token)
        if (use_mtp && mdl.mtp.loaded && total_gen == 1) {
            float verify_embd[D];
            get_embd(&mdl, prompt_tokens[n_prompt - 1], verify_embd, emb_file);
            float verify_logits[vs];
            mdl.mtp.cache_len = 0;
            wubu_mtp_draft_forward(&mdl, h_39, verify_embd, 1, verify_logits);
            int mtp_first = argmax(verify_logits, vs);
            fprintf(stderr, "\n[MTP-VFY] main predicts=%d MTP predicts=%d %s\n",
                    main_token, mtp_first, main_token == mtp_first ? "MATCH!" : "MISMATCH");
        }

        // ====== STEP 2: Generate MTP extra tokens ======
        if (use_mtp && mdl.mtp.loaded) {
            mdl.mtp.cache_len = 0;
            for (int di = 0; di < DRAFT_N; di++) {
                wubu_mtp_draft_forward(&mdl, h_39, cur, 1, mtp_logits);
                int draft_id = argmax(mtp_logits, vs);
                {
                    char piece[256];
                    int nc = wubu_tokenizer_decode(&tok, &draft_id, 1, piece, 256);
                    if (nc > 0) fwrite(piece, 1, nc, stdout); else printf("<%d>", draft_id);
                    fflush(stdout);
                }
                total_gen++;
                get_embd(&mdl, draft_id, cur, emb_file);
                if (draft_id == tok.eos_id || draft_id == tok.bos_id) break;
                if (total_gen >= max_tokens || g_stop) break;
            }
            if (total_gen >= max_tokens || g_stop) break;
        }

        // ====== STEP 3: Advance main model state ======
        // Forward last emitted token through main model → new h_39 + logits
        mdl.save_last_hidden = h_39;
        wubu_model_forward_from_embd(&mdl, cur, 1, 1, logits_out);
        mdl.save_last_hidden = NULL;
        last_logits = logits_out;
    }

    printf("\n");

    double t_total = wall_clock() - t0;
    double t_decode = t_total - t_prefill;
    fprintf(stderr, "\n--- Stats ---\n");
    fprintf(stderr, "Total: %d tok in %.2fs (%.1f tok/s)\n", total_gen + n_prompt, t_total,
           (total_gen + n_prompt) / t_total);
    fprintf(stderr, "Decode: %d tok in %.2fs (%.1f tok/s)\n", total_gen, t_decode, total_gen / t_decode);

    free(embd); free(logits_buf); free(h_next); free(logits_out); free(cur); free(mtp_logits);
    if (emb_file) fclose(emb_file);
    wubu_tokenizer_free(&tok);
    wubu_model_free(&mdl);
    return 0;
}
