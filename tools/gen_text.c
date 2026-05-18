/**
 * gen_text.c — Simple text generation using verified quantized model path.
 * Uses wubu_model_forward_from_embd for both prefill and decode.
 * SSM state carries between calls (stored in model->ssm_states).
 * GQA layers recompute attention from scratch (no KV cache yet).
 *
 * Build: make gen_text
 * Usage: ./gen_text [prompt] [max_tokens] [top_k]
 * Env: MOE=1  VERBOSE=1
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
#include <stdbool.h>

static volatile int g_stop = 0;
static void handle_sigint(int sig) { (void)sig; g_stop = 1; fprintf(stderr, "\n[interrupt]\n"); }

static double clock_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int read_embedding(const wubu_model_t *mdl, int token_id, float *out, FILE *emb_file) {
    int D = D_MODEL;
    if (mdl->use_embedding_file) {
        if (token_id >= 0 && token_id < mdl->vocab_size) {
            fseek(emb_file, (long long)token_id * D * sizeof(float), SEEK_SET);
            size_t nread = fread(out, sizeof(float), D, emb_file);
            return nread == (size_t)D ? 1 : 0;
        }
        return 0;
    } else if (mdl->token_embd && token_id >= 0 && token_id < mdl->vocab_size) {
        memcpy(out, mdl->token_embd + (long long)token_id * D, D * sizeof(float));
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *model_path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *prompt = "The meaning of life is";
    int max_tokens = 32;
    int top_k = 40;

    if (argc > 1) prompt = argv[1];
    if (argc > 2) max_tokens = atoi(argv[2]);
    if (argc > 3) top_k = atoi(argv[3]);

    signal(SIGINT, handle_sigint);

    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, model_path)) return 1;
    mdl.enable_moe = true;

    // Init tokenizer from model file
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, model_path)) {
        fprintf(stderr, "Failed to init tokenizer\n");
        wubu_model_free(&mdl);
        return 1;
    }

    int D = D_MODEL;
    int vs = mdl.vocab_size;

    // Tokenize prompt
    int prompt_tokens[512];
    int n_prompt = wubu_tokenizer_encode(&tok, prompt, prompt_tokens, 512);
    if (n_prompt <= 0) {
        fprintf(stderr, "Tokenization failed\n");
        prompt_tokens[0] = tok.bos_id >= 0 ? tok.bos_id : 248044;
        n_prompt = 1;
    }
    printf("Prompt: %d tokens\n", n_prompt);

    // Get embeddings for prompt
    float *embd = (float *)malloc(n_prompt * D * sizeof(float));
    FILE *emb_file = NULL;
    if (mdl.use_embedding_file) {
        emb_file = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (!emb_file) { fprintf(stderr, "Can't open embedding file\n"); free(embd); return 1; }
    }
    for (int i = 0; i < n_prompt; i++) {
        if (!read_embedding(&mdl, prompt_tokens[i], embd + i * D, emb_file))
            memset(embd + i * D, 0, D * sizeof(float));
    }

    // Prefill forward
    float *logits = (float *)malloc(n_prompt * vs * sizeof(float));
    double t0 = clock_seconds();
    wubu_model_forward_from_embd(&mdl, embd, 1, n_prompt, logits);
    double t_prefill = clock_seconds() - t0;

    float *last_logits = logits + (n_prompt - 1) * vs;
    int generated = 0;

    // Print prompt text
    {
        char buf[1024];
        int nc = wubu_tokenizer_decode(&tok, prompt_tokens, n_prompt, buf, 1024);
        if (nc > 0) printf("Input: %s\n", buf);
    }

    // Decode loop
    while (generated < max_tokens && !g_stop) {
        // Top-k + greedy
        int topk_idxs[256];
        int nk = top_k > 256 ? 256 : top_k;
        for (int k = 0; k < nk; k++) {
            float maxv = -1e30f; int maxi = -1;
            for (int i = 0; i < vs; i++)
                if (last_logits[i] > maxv) { maxv = last_logits[i]; maxi = i; }
            topk_idxs[k] = maxi;
            last_logits[maxi] = -1e30f;
        }
        int next_token = topk_idxs[0];

        char piece_buf[256];
        int n_chars = wubu_tokenizer_decode(&tok, &next_token, 1, piece_buf, 256);
        if (n_chars > 0) fwrite(piece_buf, 1, n_chars, stdout);
        else printf("<%d>", next_token);
        fflush(stdout);

        if (next_token == tok.eos_id || next_token == tok.bos_id) break;

        // Get next embedding
        float x_next[D_MODEL];
        if (!read_embedding(&mdl, next_token, x_next, emb_file))
            memset(x_next, 0, D_MODEL * sizeof(float));

        wubu_model_forward_from_embd(&mdl, x_next, 1, 1, logits);
        last_logits = logits;
        generated++;
    }
    printf("\n");

    // Stats
    double t_total = clock_seconds() - t0;
    printf("\n--- Stats ---\n");
    printf("Prefill: %d tok in %.2fs (%.1f tok/s)\n", n_prompt, t_prefill, n_prompt / t_prefill);
    double t_decode = t_total - t_prefill;
    if (generated > 0 && t_decode > 0)
        printf("Decode:  %d tok in %.2fs (%.1f tok/s)\n", generated, t_decode, generated / t_decode);

    free(logits);
    free(embd);
    if (emb_file) fclose(emb_file);
    wubu_tokenizer_free(&tok);
    wubu_model_free(&mdl);
    return 0;
}
