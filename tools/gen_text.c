/**
 * gen_text.c — Text generation with optional GPU-accelerated output projection.
 *
 * CPU-only:  make gen_text
 * GPU:       GPU=1 GPU_BATCH=16 OMP_NUM_THREADS=16 make gen_text_gpu
 *
 * Environment:
 *   GPU=1       — Enable GPU output projection
 *   GPU_BATCH=N — Max batch size for batched prefill (default 1)
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

// GPU support — compiled only in gen_text_gpu target (-DGPU_SUPPORT)
#ifdef GPU_SUPPORT
#include "gpu_output_proj.h"
#else
static inline bool gpu_output_init(const void *w,int D,int V,int t){(void)w;(void)D;(void)V;(void)t;return false;}
static inline bool gpu_output_project_batch(const float *i,float *o,int T){(void)i;(void)o;(void)T;return false;}
static inline bool gpu_output_project(const float *i,float *o){(void)i;(void)o;return false;}
static inline void gpu_output_cleanup(void){}
inline int wubu_model_gpu_init(wubu_model_t *m,int mc,int cs){(void)m;(void)mc;(void)cs;return 0;}
inline void wubu_model_gpu_free(wubu_model_t *m){(void)m;}
#endif

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

    // GPU init (if GPU=1 env var set)
    int use_gpu = getenv("GPU") != NULL;
    if (use_gpu) {
        // Initialize integrated GPU context: GQA layers, KV cache, chunked attention
        int max_ctx = getenv("MAX_CTX") ? atoi(getenv("MAX_CTX")) : 262144;
        int chunk_sz = getenv("GPU_CHUNK") ? atoi(getenv("GPU_CHUNK")) : 256;
        if (!wubu_model_gpu_init(&mdl, max_ctx, chunk_sz)) {
            fprintf(stderr, "GPU GQA init failed, falling back to CPU GQA\n");
        } else {
            printf("GPU: GQA acceleration active (max_ctx=%d, chunk=%d)\n", max_ctx, chunk_sz);
        }
        // Also init GPU output projection (existing quantized Q4_K path)
        if (!gpu_output_init(mdl.output_weight_q, D_MODEL, mdl.vocab_size, mdl.output_weight_type)) {
            fprintf(stderr, "GPU output proj init failed, falling back to CPU\n");
            use_gpu = 0;
        }
    }

    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, model_path)) {
        fprintf(stderr, "Failed to init tokenizer\n");
        wubu_model_free(&mdl);
        return 1;
    }

    int D = D_MODEL;
    int vs = mdl.vocab_size;

    // Tokenize prompt
    int prompt_tokens[1024];
    int n_prompt;
    int chat_mode = getenv("CHAT") != NULL;
    if (chat_mode) {
        const int IM_START = 248045, IM_END = 248046, THINK = 248068, NL_TOKEN = 198;
        int pos = 0;
        prompt_tokens[pos++] = tok.bos_id;
        prompt_tokens[pos++] = IM_START;
        int n = wubu_tokenizer_encode(&tok, "system\nYou are a helpful assistant.", prompt_tokens + pos, 1024 - pos);
        if (n <= 0) return 1; pos += n;
        prompt_tokens[pos++] = IM_END; prompt_tokens[pos++] = NL_TOKEN;
        prompt_tokens[pos++] = IM_START;
        n = wubu_tokenizer_encode(&tok, "user\n", prompt_tokens + pos, 1024 - pos);
        if (n <= 0) return 1; pos += n;
        n = wubu_tokenizer_encode(&tok, prompt, prompt_tokens + pos, 1024 - pos);
        if (n <= 0) return 1; pos += n;
        prompt_tokens[pos++] = IM_END; prompt_tokens[pos++] = NL_TOKEN;
        prompt_tokens[pos++] = IM_START;
        n = wubu_tokenizer_encode(&tok, "assistant\n", prompt_tokens + pos, 1024 - pos);
        if (n <= 0) return 1; pos += n;
        prompt_tokens[pos++] = THINK; prompt_tokens[pos++] = NL_TOKEN;
        n_prompt = pos;
    } else {
        n_prompt = wubu_tokenizer_encode(&tok, prompt, prompt_tokens, 1024);
        if (n_prompt <= 0) { prompt_tokens[0] = tok.bos_id >= 0 ? tok.bos_id : 248044; n_prompt = 1; }
    }
    printf("Prompt: %d tokens\n", n_prompt);

    // Embeddings
    float *embd = (float *)malloc(n_prompt * D * sizeof(float));
    FILE *emb_file = NULL;
    if (mdl.use_embedding_file) {
        emb_file = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        if (!emb_file) { free(embd); return 1; }
    }
    for (int i = 0; i < n_prompt; i++)
        if (!read_embedding(&mdl, prompt_tokens[i], embd + i * D, emb_file))
            memset(embd + i * D, 0, D * sizeof(float));

    // Prefill: logits or hidden states
    float *logits = (float *)malloc(n_prompt * vs * sizeof(float));
    double t0 = clock_seconds();

    if (use_gpu) {
        // GPU path: forward saves hidden states, GPU does output proj
        mdl.skip_output_proj = true;
        wubu_model_forward_from_embd(&mdl, embd, 1, n_prompt, logits);
        // logits now has n_prompt * D_MODEL floats (hidden states at vs stride)
        // Pack them contiguously for batched SGEMM
        float *hidden_batch = (float *)malloc(n_prompt * D * sizeof(float));
        for (int i = 0; i < n_prompt; i++)
            memcpy(hidden_batch + i * D, logits + i * vs, D * sizeof(float));
        // Batched GPU output projection
        if (n_prompt > 0)
            gpu_output_project_batch(hidden_batch, logits, n_prompt);
        free(hidden_batch);
    } else {
        mdl.skip_output_proj = false;
        wubu_model_forward_from_embd(&mdl, embd, 1, n_prompt, logits);
    }

    double t_prefill = clock_seconds() - t0;

    float *last_logits = logits + (n_prompt - 1) * vs;
    int generated = 0;

    { char buf[1024]; int nc = wubu_tokenizer_decode(&tok, prompt_tokens, n_prompt, buf, 1024);
      if (nc > 0) printf("Input: %s\n", buf); }

    // Decode loop
    while (generated < max_tokens && !g_stop) {
        int topk_idxs[256];
        int nk = top_k > 256 ? 256 : top_k;
        for (int k = 0; k < nk; k++) {
            float maxv = -1e30f; int maxi = -1;
            for (int i = 0; i < vs; i++)
                if (last_logits[i] > maxv) { maxv = last_logits[i]; maxi = i; }
            topk_idxs[k] = maxi; last_logits[maxi] = -1e30f;
        }
        int next_token = topk_idxs[0];

        char piece_buf[256];
        int n_chars = wubu_tokenizer_decode(&tok, &next_token, 1, piece_buf, 256);
        if (n_chars > 0) fwrite(piece_buf, 1, n_chars, stdout);
        else printf("<%d>", next_token);
        fflush(stdout);

        if (next_token == tok.eos_id || next_token == tok.bos_id) break;

        float x_next[D_MODEL];
        if (!read_embedding(&mdl, next_token, x_next, emb_file))
            memset(x_next, 0, D_MODEL * sizeof(float));

        if (use_gpu) {
            mdl.skip_output_proj = true;
            wubu_model_forward_from_embd(&mdl, x_next, 1, 1, logits);
            gpu_output_project(logits, logits);
        } else {
            mdl.skip_output_proj = false;
            wubu_model_forward_from_embd(&mdl, x_next, 1, 1, logits);
        }
        last_logits = logits;
        generated++;
    }
    printf("\n");

    double t_total = clock_seconds() - t0;
    printf("\n--- Stats ---\n");
    printf("Prefill: %d tok in %.2fs (%.1f tok/s)\n", n_prompt, t_prefill, n_prompt / t_prefill);
    double t_decode = t_total - t_prefill;
    if (generated > 0 && t_decode > 0)
        printf("Decode:  %d tok in %.2fs (%.1f tok/s)\n", generated, t_decode, generated / t_decode);

    free(logits); free(embd);
    if (emb_file) fclose(emb_file);
    wubu_tokenizer_free(&tok);
    gpu_output_cleanup();
    wubu_model_free(&mdl);
    return 0;
}
