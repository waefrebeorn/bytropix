/**
 * gen_text_mtp.c — Multi-Token Prediction inference.
 * Uses MTP head as free extra token generator.
 * Each decode step: 1 main model forward + DRAFT_N MTP head forwards.
 * No verify/rollback — MTP outputs are treated as model's own predictions.
 *
 * NOTE: At IQ2_M quantization, MTP head predictions may differ significantly
 * from main model's. This is a known limitation of quantized MTP.
 *
 * Build: make gen_text_mtp
 * Usage: MOE=1 ./gen_text_mtp [prompt] [max_tokens]
 *        NO_MTP=1 ./gen_text_mtp [prompt] [max_tokens]  (disable MTP)
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
    int use_mtp = getenv("MTP") != NULL;  // opt-in: requires MTP=1

    if (argc > 1) prompt = argv[1];
    if (argc > 2) max_tokens = atoi(argv[2]);

    signal(SIGINT, handle_sigint);

    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, model_path)) return 1;
    mdl.enable_moe = true;

    if (use_mtp) {
        // Load MTP head from same GGUF
        if (!wubu_mtp_load(&mdl.mtp, model_path, mdl.gguf_ctx, (const uint8_t*)mdl.gguf_ctx->data_blob)) {
            fprintf(stderr, "MTP head not available, falling back to non-MTP\n");
            use_mtp = 0;
        } else {
            fprintf(stderr, "MTP head loaded (opt-in mode)\n");
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

    // Use the prefill result
    float *last_logits = logits_buf + (n_prompt - 1) * vs;
    int last_real_token = argmax(last_logits, vs);
    get_embd(&mdl, last_real_token, cur, emb_file);

    while (total_gen < max_tokens && !g_stop) {
        // Emit main model's prediction
        {
            char piece[256];
            int nc = wubu_tokenizer_decode(&tok, &last_real_token, 1, piece, 256);
            if (nc > 0) fwrite(piece, 1, nc, stdout); else printf("<%d>", last_real_token);
            fflush(stdout);
        }
        total_gen++;
        if (last_real_token == tok.eos_id || last_real_token == tok.bos_id) break;
        if (total_gen >= max_tokens || g_stop) break;

        // Generate and emit MTP extra tokens
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

        // Advance main model state by forwarding the last emitted token
        // This brings KV cache + SSM state in sync with generated text
        mdl.save_last_hidden = h_39;
        wubu_model_forward_from_embd(&mdl, cur, 1, 1, logits_out);
        mdl.save_last_hidden = NULL;
        last_real_token = argmax(logits_out, vs);
        get_embd(&mdl, last_real_token, cur, emb_file);
    }

    printf("\n");

    double t_total = wall_clock() - t0;
    double t_decode = t_total - t_prefill;
    fprintf(stderr, "\n--- Stats ---\n");
    fprintf(stderr, "Total: %d tok in %.2fs (%.1f tok/s)\n", total_gen + n_prompt, t_total,
           (total_gen + n_prompt) / t_total);
    fprintf(stderr, "Decode: %d tok in %.2fs (%.1f tok/s)\n", total_gen, t_decode, total_gen / t_decode);
    if (mdl.mtp.loaded) {
        int main_forwards = total_gen - (!no_mtp ? (total_gen / (DRAFT_N + 1)) : 0);
        if (main_forwards < 1) main_forwards = 1;
        fprintf(stderr, "MTP enabled: ~%d main forwards\n", total_gen / (no_mtp ? 1 : (DRAFT_N + 1)));
    }

    free(embd); free(logits_buf); free(h_next); free(logits_out); free(cur); free(mtp_logits);
    if (emb_file) fclose(emb_file);
    wubu_tokenizer_free(&tok);
    wubu_model_free(&mdl);
    return 0;
}
