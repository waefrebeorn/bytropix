/**
 * gen_text_mtp.c — Clean decode with model->save_last_hidden for h_39 capture.
 * MTP draft + verify commented out — needs SSM state save/restore.
 * 
 * Build: make gen_text_mtp
 * Usage: MOE=1 ./gen_text_mtp [prompt] [max_tokens]
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
    // Match gen_text.c precedence: emb_file first if use_embedding_file
    if (mdl->use_embedding_file && emb_file) {
        fseek(emb_file, (long long)token * D_MODEL * sizeof(float), SEEK_SET);
        size_t n = fread(out, sizeof(float), D_MODEL, emb_file);
        if (n != (size_t)D_MODEL) memset(out, 0, D_MODEL * sizeof(float));
    } else if (mdl->token_embd) {
        memcpy(out, mdl->token_embd + (long long)token * D_MODEL, D_MODEL * sizeof(float));
    } else {
        memset(out, 0, D_MODEL * sizeof(float));
    }
}

int main(int argc, char **argv) {
    const char *model_path = "/models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf";
    const char *prompt = "The meaning of life is";
    int max_tokens = 32;
    int D = D_MODEL;

    if (argc > 1) prompt = argv[1];
    if (argc > 2) max_tokens = atoi(argv[2]);

    signal(SIGINT, handle_sigint);

    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, model_path)) return 1;
    mdl.enable_moe = true;

    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, model_path)) { wubu_model_free(&mdl); return 1; }

    int vs = mdl.vocab_size;
    int prompt_tokens[1024];
    int n_prompt = wubu_tokenizer_encode(&tok, prompt, prompt_tokens, 1024);
    if (n_prompt <= 0) { prompt_tokens[0] = tok.bos_id >= 0 ? tok.bos_id : 248044; n_prompt = 1; }
    printf("Prompt: %d tokens\n", n_prompt);

    float *embd = (float *)malloc(n_prompt * D * sizeof(float));
    FILE *emb_file = NULL;
    if (mdl.use_embedding_file) {
        emb_file = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (!emb_file) { free(embd); return 1; }
    }
    for (int i = 0; i < n_prompt; i++)
        get_embd(&mdl, prompt_tokens[i], embd + i * D, emb_file);

    // Prefill: get logits + h_39 via save_last_hidden
    float *logits_buf = (float *)malloc(n_prompt * vs * sizeof(float));
    float *h_39 = (float *)malloc(D * sizeof(float));
    double t0 = wall_clock();
    mdl.save_last_hidden = h_39;
    wubu_model_forward_from_embd(&mdl, embd, 1, n_prompt, logits_buf);
    mdl.save_last_hidden = NULL;
    double t_prefill = wall_clock() - t0;
    fprintf(stderr, "Prefill: %.2fs\n", t_prefill);

    // Print prompt
    { char buf[2048]; int nc = wubu_tokenizer_decode(&tok, prompt_tokens, n_prompt, buf, 2048);
      if (nc > 0) printf("Input: %s\n", buf); }

    float *logits_out = (float *)malloc(vs * sizeof(float));
    float *h_next = (float *)malloc(D * sizeof(float));
    int total_gen = 0;

    while (total_gen < max_tokens && !g_stop) {
        int next_token;
        if (total_gen == 0) {
            // First iteration: use prefill results
            float *last_prompt_logits = logits_buf + (n_prompt - 1) * vs;
            next_token = argmax(last_prompt_logits, vs);
        } else {
            next_token = argmax(logits_out, vs);
        }

        char piece[256];
        int nc = wubu_tokenizer_decode(&tok, &next_token, 1, piece, 256);
        if (nc > 0) fwrite(piece, 1, nc, stdout); else printf("<%d>", next_token);
        fflush(stdout);
        if (next_token == tok.eos_id || next_token == tok.bos_id) break;

        total_gen++;
        if (total_gen >= max_tokens || g_stop) break;

        // Forward ground truth token → get h_next (for next iteration's MTP)
        float embd_next[D];
        get_embd(&mdl, next_token, embd_next, emb_file);

        mdl.save_last_hidden = h_next;
        wubu_model_forward_from_embd(&mdl, embd_next, 1, 1, logits_out);
        mdl.save_last_hidden = NULL;
    }
    printf("\n");

    double t_total = wall_clock() - t0;
    printf("\n--- Stats ---\n");
    printf("Total: %d tok in %.2fs (%.1f tok/s)\n", total_gen + n_prompt, t_total,
           (total_gen + n_prompt) / t_total);
    if (total_gen > 0) {
        double t_decode = t_total - t_prefill;
        printf("Decode: %d tok in %.2fs (%.1f tok/s)\n", total_gen, t_decode, total_gen / t_decode);
    }

    free(embd); free(logits_buf); free(h_39); free(h_next); free(logits_out);
    if (emb_file) fclose(emb_file);
    wubu_tokenizer_free(&tok);
    wubu_model_free(&mdl);
    return 0;
}
