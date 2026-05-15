/**
 * infer_text.c — Full text generation pipeline
 *
 * Loads tokenizer + model + embeddings from GGUF.
 * Runs autoregressive generation with MoE.
 *
 * Usage: ./infer_text [gguf_path] ["prompt text"] [max_tokens] [top_k]
 *
 * Env: MOE=1 enable MoE (default: 0)
 *      MOE_LAYERS=N limit MoE to first N layers (0=all)
 *      VERBOSE=1 layer-by-layer timing
 */
#include "wubu_model.h"
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include "wubu_tokenizer.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Load token_embd.weight from GGUF
static float *load_embeddings(gguf_ctx *ctx, int *vocab_size) {
    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t) { fprintf(stderr, "token_embd.weight not found\n"); return NULL; }
    int64_t n_elems = 1;
    for (int i = 0; i < t->n_dims; i++) n_elems *= t->dims[i];
    int vs = (int)(n_elems / D_MODEL);
    float *embd = (float *)malloc(n_elems * sizeof(float));
    if (!embd) return NULL;
    if (gguf_read_tensor_f32(ctx, t, embd, n_elems) <= 0) {
        free(embd); return NULL;
    }
    *vocab_size = vs;
    printf("  Embeddings: %d tokens from GGUF (%ld MB)\n", vs, n_elems * sizeof(float) / (1024*1024));
    return embd;
}

// Greedy sample: argmax
static int sample_greedy(const float *logits, int vocab_size) {
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return best;
}

// Top-K sample
static int sample_topk(const float *logits, int vocab_size, int k) {
    if (k <= 1) return sample_greedy(logits, vocab_size);
    typedef struct { float val; int idx; } pair_t;
    pair_t *pairs = (pair_t *)malloc(vocab_size * sizeof(pair_t));
    for (int i = 0; i < vocab_size; i++) { pairs[i].val = logits[i]; pairs[i].idx = i; }
    int kk = k < vocab_size ? k : vocab_size;
    for (int i = 0; i < kk; i++) {
        int best = i;
        for (int j = i+1; j < vocab_size; j++)
            if (pairs[j].val > pairs[best].val) best = j;
        pair_t tmp = pairs[i]; pairs[i] = pairs[best]; pairs[best] = tmp;
    }
    float max_v = pairs[0].val;
    float sum = 0;
    for (int i = 0; i < kk; i++) sum += (float)exp((double)(pairs[i].val - max_v));
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0;
    for (int i = 0; i < kk; i++) {
        cum += (float)exp((double)(pairs[i].val - max_v)) / sum;
        if (r <= cum) { int idx = pairs[i].idx; free(pairs); return idx; }
    }
    int idx = pairs[kk-1].idx; free(pairs); return idx;
}

static volatile int stop_requested = 0;
static void sigint_handler(int sig) { (void)sig; stop_requested = 1; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
        : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *prompt = argc > 2 ? argv[2] : "The meaning of life is";
    int max_tokens = argc > 3 ? atoi(argv[3]) : 64;
    int top_k = argc > 4 ? atoi(argv[4]) : 1;
    int verbose = 0;
    if (getenv("VERBOSE")) verbose = atoi(getenv("VERBOSE"));

    signal(SIGINT, sigint_handler);

    printf("=== WuBuText AI — Text Inference ===\n");
    printf("Model: %s\n", path);
    printf("Prompt: \"%s\"\n", prompt);
    printf("Max tokens: %d | Sampling: %s\n", max_tokens, top_k <= 1 ? "greedy" : "topk-%d");

    double t_total = now_sec();

    // ===== 1. Load GGUF =====
    double t0 = now_sec();
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    printf("GGUF load: %.2f s\n", now_sec() - t0);

    // ===== 2. Load Tokenizer =====
    t0 = now_sec();
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, path)) {
        fprintf(stderr, "Tokenizer init failed\n"); gguf_close(ctx); return 1;
    }
    printf("Tokenizer: %d tokens, %d merges (%.2f s)\n", tok.vocab_size, tok.n_merges, now_sec() - t0);

    // ===== 3. Load Model =====
    t0 = now_sec();
    wubu_model_t model;
    if (!wubu_model_init(&model, path)) {
        fprintf(stderr, "Model init failed\n"); wubu_tokenizer_free(&tok); gguf_close(ctx); return 1;
    }
    if (model.gguf_ctx) { gguf_close(model.gguf_ctx); }
    model.gguf_ctx = ctx;
    printf("Model init: %.2f s\n", now_sec() - t0);

    // ===== 4. Load Embeddings =====
    t0 = now_sec();
    int vocab_size;
    float *token_embd = load_embeddings(ctx, &vocab_size);
    if (!token_embd) {
        fprintf(stderr, "Failed to load embeddings\n");
        wubu_model_free(&model); wubu_tokenizer_free(&tok); return 1;
    }
    printf("Embeddings: %.2f s\n", now_sec() - t0);

    // ===== 5. MoE quant metadata =====
    int moe_enabled = 0;
    if (getenv("MOE")) moe_enabled = atoi(getenv("MOE"));
    int moe_max_layers = 0;
    if (getenv("MOE_LAYERS")) moe_max_layers = atoi(getenv("MOE_LAYERS"));
    typedef struct { bool has_moe; } moe_info_t;
    moe_info_t *moe_info = NULL;
    if (moe_enabled) {
        moe_info = (moe_info_t *)calloc(model.n_layers, sizeof(moe_info_t));
        if (moe_info) {
            for (int l = 0; l < model.n_layers; l++) {
                char name[256];
                snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
                gguf_tensor_info *t = gguf_find_tensor(ctx, name);
                moe_info[l].has_moe = (t != NULL);
            }
        }
    }

    // ===== 6. Encode Prompt =====
    int prompt_ids[65536];
    int n_prompt = wubu_tokenizer_encode(&tok, prompt, prompt_ids, 65536);
    if (n_prompt <= 0) {
        fprintf(stderr, "Tokenization failed (%d)\n", n_prompt);
        free(token_embd); free(moe_info);
        wubu_model_free(&model); wubu_tokenizer_free(&tok); return 1;
    }
    printf("Prompt: %d tokens\n", n_prompt);

    // ===== 7. Generation Loop =====
    int max_buf = n_prompt + max_tokens + 1;
    int *all_ids = (int *)malloc(max_buf * sizeof(int));
    memcpy(all_ids, prompt_ids, n_prompt * sizeof(int));

    // Print prompt
    char out_buf[1048576];
    int out_len = wubu_tokenizer_decode(&tok, all_ids, n_prompt, out_buf, 1048576);
    if (out_len > 0) { out_buf[out_len] = '\0'; printf("%s", out_buf); fflush(stdout); }

    float *logits = (float *)malloc(vocab_size * sizeof(float));

    double t_gen_start = now_sec();
    int gen_tokens = 0;

    for (int pos = n_prompt; pos < max_buf - 1; pos++) {
        if (stop_requested) break;

        int cur_T = pos;  // total tokens in context
        int N = cur_T;

        // Build embeddings for all tokens in context
        float *embd_buf = (float *)malloc(N * D_MODEL * sizeof(float));
        for (int i = 0; i < N; i++) {
            int tid = all_ids[i];
            if (tid >= 0 && tid < vocab_size) {
                memcpy(embd_buf + i * D_MODEL, token_embd + tid * D_MODEL, D_MODEL * sizeof(float));
            } else {
                memset(embd_buf + i * D_MODEL, 0, D_MODEL * sizeof(float));
            }
        }

        // Reset SSM states for clean forward (sequential inference)
        memset(model.ssm_states, 0,
               model.n_layers * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
        memset(model.conv_states, 0,
               model.n_layers * (CONV_KERNEL - 1) * CONV_DIM * sizeof(float));

        // Allocate forward buffers
        float *x = (float *)malloc(N * D_MODEL * sizeof(float));
        float *normed = (float *)malloc(N * D_MODEL * sizeof(float));
        float *attn_out = (float *)malloc(N * D_MODEL * sizeof(float));
        float *normed2 = (float *)malloc(N * D_MODEL * sizeof(float));
        float *ffn_out = (float *)malloc(N * D_MODEL * sizeof(float));

        memcpy(x, embd_buf, N * D_MODEL * sizeof(float));

        // Forward pass through all layers
        for (int l = 0; l < model.n_layers; l++) {
            wubu_layer_t *layer = &model.layers[l];

            wubu_rms_norm(1, cur_T, D_MODEL, x, layer->attn_norm_weight, 1e-6f, normed);

            if (layer->is_ssm) {
                float *ss = model.ssm_states + l * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
                float *cs = model.conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
                wubu_ssm_forward(normed, 1, cur_T, &layer->ssm, ss, cs, attn_out);
            } else {
                wubu_gqa_forward(normed, 1, cur_T, &layer->gqa, attn_out);
            }

            // NaN check
            if (verbose) {
                int has_nan = 0;
                for (int i = 0; i < N * D_MODEL; i++)
                    if (isnan(attn_out[i])) { has_nan = 1; break; }
                if (has_nan) printf("[L%d NaN] ", l);
            }

            for (int i = 0; i < N * D_MODEL; i++) x[i] += attn_out[i];
            wubu_rms_norm(1, cur_T, D_MODEL, x, layer->post_attn_norm_weight, 1e-6f, normed2);

            if (moe_enabled && moe_info && moe_info[l].has_moe &&
                (moe_max_layers == 0 || l < moe_max_layers)) {
                if (wubu_moe_load_layer(ctx, l, &layer->moe)) {
                    wubu_moe_forward(normed2, 1, cur_T, &layer->moe, ffn_out);
                    wubu_moe_free_layer(&layer->moe);
                } else {
                    memcpy(ffn_out, normed2, N * D_MODEL * sizeof(float));
                }
            } else {
                memcpy(ffn_out, normed2, N * D_MODEL * sizeof(float));
            }

            for (int i = 0; i < N * D_MODEL; i++) x[i] += ffn_out[i];
        }

        // Final RMSNorm
        if (model.norm_weight) {
            wubu_rms_norm(1, cur_T, D_MODEL, x, model.norm_weight, 1e-6f, normed);
            memcpy(x, normed, N * D_MODEL * sizeof(float));
        }

        // Output projection: last token
        const float *h_last = x + (cur_T - 1) * D_MODEL;
        if (model.output_weight) {
            for (int j = 0; j < vocab_size; j++) {
                double sum = 0.0;
                for (int k = 0; k < D_MODEL; k++)
                    sum += (double)h_last[k] * (double)model.output_weight[j * D_MODEL + k];
                logits[j] = (float)sum;
            }
        } else {
            memcpy(logits, h_last, D_MODEL * sizeof(float));
        }

        // Sample
        int next_id = sample_topk(logits, vocab_size, top_k);
        all_ids[pos] = next_id;
        gen_tokens++;

        // Decode ALL tokens and find the new piece
        int new_out_len = wubu_tokenizer_decode(&tok, all_ids, pos + 1, out_buf, 1048576);
        if (new_out_len > out_len) {
            out_buf[new_out_len] = '\0';
            printf("%s", out_buf + out_len);
            fflush(stdout);
            out_len = new_out_len;
        }

        free(x); free(normed); free(attn_out); free(normed2); free(ffn_out);
        free(embd_buf);

        if (gen_tokens >= max_tokens) break;
        if (next_id == tok.eos_id && gen_tokens > 1) {
            printf("\n[EOS]");
            break;
        }
    }

    double t_gen = now_sec() - t_gen_start;
    printf("\n\n=== Generation Complete ===\n");
    printf("Generated %d tokens in %.2f s (%.1f tok/s)\n",
           gen_tokens, t_gen, gen_tokens / t_gen);
    printf("Total time: %.2f s\n", now_sec() - t_total);

    // ===== 8. Cleanup =====
    free(token_embd);
    free(logits);
    free(all_ids);
    free(moe_info);
    wubu_model_free(&model);
    wubu_tokenizer_free(&tok);

    printf("=== Text Inference PASS ===\n");
    return 0;
}
