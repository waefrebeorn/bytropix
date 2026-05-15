/**
 * test_regression.c — Regression test suite comparing bytropix engine output
 * against llama.cpp reference on 10 fixed prompts.
 *
 * For each prompt:
 *   1. Tokenizes the prompt (same vocab)
 *   2. Looks up embeddings from GGUF
 *   3. Runs wubu_model_forward_from_embd → gets our top-5 predicted tokens
 *   4. Shells out to llama-cli (greedy, temp=0.0, 1 token) → gets reference top token
 *   5. Reports comparison: top-3 from both, top-1 match, Jaccard(top5_us, {ref}+our_top4)
 *
 * Compilation template:
 *   cd /home/wubu/bytropix && make tools/test_regression
 *
 * Requires Makefile entry:
 *   test_regression: tools/test_regression.c $(MODEL_OBJ) src/wubu_tokenizer.o $(CUDA_OBJ)
 *       $(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda/lib64 -lstdc++
 */
#include "wubu_model.h"
#include "gguf_reader.h"
#include "wubu_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_PROMPTS 10
#define MAX_TOKENS 2048
#define TOP_K 5
#define MAX_CMD 8192
#define MAX_OUTPUT 16384

// ─────────────────────────────────────────────
// 10 fixed prompts (exactly as specified)
// ─────────────────────────────────────────────
static const char *prompts[MAX_PROMPTS] = {
    "Hello! How are you?",
    "The capital of France is",
    "Once upon a time",
    "2 + 2 =",
    "Translate to French: Hello",
    "What is quantum computing?",
    "Write a poem about AI",
    "Explain gravity simply",
    "Python code for fibonacci",
    "Summary: The quick brown fox"
};

// ─────────────────────────────────────────────
// Helpers: find top-K indices in logits
// ─────────────────────────────────────────────
static int find_top_k(const float *logits, int vocab_size, int k, int *top_ids, float *top_vals) {
    if (k > vocab_size) k = vocab_size;
    for (int i = 0; i < k; i++) {
        top_ids[i] = i;
        top_vals[i] = logits[i];
    }
    // Initialize with first k
    for (int i = k; i < vocab_size; i++) {
        float v = logits[i];
        if (v > top_vals[k-1]) {
            top_vals[k-1] = v;
            top_ids[k-1] = i;
            // Bubble up
            for (int j = k-2; j >= 0; j--) {
                if (top_vals[j] < top_vals[j+1]) {
                    float tmpf = top_vals[j]; top_vals[j] = top_vals[j+1]; top_vals[j+1] = tmpf;
                    int tmpi = top_ids[j]; top_ids[j] = top_ids[j+1]; top_ids[j+1] = tmpi;
                } else break;
            }
        }
    }
    return k;
}

// ─────────────────────────────────────────────
// Run llama-cli as subprocess, capture output text
// ─────────────────────────────────────────────
static int run_llamacpp(const char *prompt, char *output_buf, int max_out) {
    // Write prompt to temp file to avoid shell escaping issues
    char tmpfile[] = "/tmp/test_reg_prompt_XXXXXX";
    int fd = mkstemp(tmpfile);
    if (fd < 0) return -1;
    FILE *fp = fdopen(fd, "w");
    if (!fp) { close(fd); return -1; }
    fprintf(fp, "%s", prompt);
    fclose(fp);

    // Build command: llama-cli with greedy decoding, 1 token, no display
    char cmd[MAX_CMD];
    snprintf(cmd, sizeof(cmd),
        "%s -m %s -f %s -n 1 --temp 0.0 --simple-io --no-display-prompt -c 4096 2>/dev/null",
        "/home/wubu/llama.cpp/build/bin/llama-cli",
        "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf",
        tmpfile);

    FILE *pipe = popen(cmd, "r");
    if (!pipe) {
        unlink(tmpfile);
        return -1;
    }

    size_t total = 0;
    char line[4096];
    int in_body = 0;  // skip the banner/ASCII art
    while (fgets(line, sizeof(line), pipe)) {
        // Skip banner lines: start with block chars or "Loading" or "build" or empty
        if (!in_body) {
            // Skip until we see the prompt echoed back or the actual response
            if (strstr(line, "> ") || strstr(line, "[Prompt:") || 
                strstr(line, "Loading") || strstr(line, "build") ||
                strstr(line, "available") || strstr(line, "modalities") ||
                strstr(line, "\xe2\x96\x84") || strlen(line) <= 1) {
                continue;
            }
            // Skip known repeating patterns
            if (strstr(line, "llama-cli")) continue;
            if (strstr(line, "EXIT")) continue;
            in_body = 1;
        }
        // Collect non-garbage output (skip empty lines and prompt markers)
        if (strlen(line) <= 1) continue;
        if (strstr(line, "> ")) continue;
        if (strstr(line, "[Prompt:")) continue;
        if (strstr(line, "[Generation:")) continue;
        
        size_t len = strlen(line);
        if (total + len + 1 < (size_t)max_out) {
            memcpy(output_buf + total, line, len);
            total += len;
            output_buf[total] = '\0';
        }
    }
    int status = pclose(pipe);
    unlink(tmpfile);
    return (total > 0) ? (int)total : 0;
}

// ─────────────────────────────────────────────
// Jaccard similarity of two integer sets (each size k)
// ─────────────────────────────────────────────
static float jaccard(const int *set1, const int *set2, int k) {
    int intersect = 0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            if (set1[i] == set2[j]) {
                intersect++;
                break;
            }
        }
    }
    int union_size = 2 * k - intersect;
    return (union_size > 0) ? (float)intersect / union_size : 0.0f;
}

// ─────────────────────────────────────────────
// Check if a token ID is present in a set
// ─────────────────────────────────────────────
static int in_set(int id, const int *set, int k) {
    for (int i = 0; i < k; i++)
        if (set[i] == id) return 1;
    return 0;
}

// ─────────────────────────────────────────────
// Scrub non-printable bytes from a string for safe display
// ─────────────────────────────────────────────
static void scrub(char *s) {
    for (; *s; s++) {
        if (*s < 32 && *s != '\n' && *s != '\t') *s = '?';
    }
}

// ─────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────
int main() {
    const char *model_path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *llama_cli  = "/home/wubu/llama.cpp/build/bin/llama-cli";
    const int D = 2048;  // D_MODEL

    printf("=== BYTROPIX vs llama.cpp REGRESSION TEST ===\n");
    printf("Model: %s\n", model_path);
    printf("llama-cli: %s\n\n", llama_cli);

    // ── Initialize our model ──
    wubu_model_t model;
    if (!wubu_model_init(&model, model_path)) {
        fprintf(stderr, "FAILED to initialize bytropix model\n");
        return 1;
    }

    // ── Initialize tokenizer ──
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, model_path)) {
        fprintf(stderr, "FAILED to initialize tokenizer\n");
        wubu_model_free(&model);
        return 1;
    }

    // ── Load token embeddings from GGUF ──
    gguf_ctx *ctx = model.gguf_ctx;
    gguf_tensor_info *t = gguf_find_tensor(ctx, "token_embd.weight");
    if (!t) {
        fprintf(stderr, "FAILED to find token_embd.weight in GGUF\n");
        wubu_tokenizer_free(&tok);
        wubu_model_free(&model);
        return 1;
    }
    int64_t ne = t->dims[0] * t->dims[1];
    int vocab_size = (int)(ne / D);
    float *embd = (float *)malloc(ne * sizeof(float));
    if (!embd) {
        fprintf(stderr, "FAILED to allocate embedding buffer\n");
        wubu_tokenizer_free(&tok);
        wubu_model_free(&model);
        return 1;
    }
    int read_ok = gguf_read_tensor_f32(ctx, t, embd, ne);
    if (read_ok == 0) {
        fprintf(stderr, "FAILED to read token_embd.weight\n");
        free(embd);
        wubu_tokenizer_free(&tok);
        wubu_model_free(&model);
        return 1;
    }
    printf("Vocabulary size: %d\n\n", vocab_size);

    // ── Allocate per-prompt buffers ──
    int *pids = (int *)malloc(MAX_TOKENS * sizeof(int));
    float *x   = (float *)malloc(MAX_TOKENS * D * sizeof(float));
    float *logits = (float *)malloc(MAX_TOKENS * vocab_size * sizeof(float));
    if (!pids || !x || !logits) {
        fprintf(stderr, "FAILED to allocate working buffers\n");
        free(embd); free(pids); free(x); free(logits);
        wubu_tokenizer_free(&tok);
        wubu_model_free(&model);
        return 1;
    }

    // Results tracking
    int total_matches = 0, total_prompts = 0;
    float total_jaccard = 0.0f;
    char ref_output[MAX_OUTPUT];

    // ── Loop over prompts ──
    for (int pi = 0; pi < MAX_PROMPTS; pi++) {
        const char *prompt = prompts[pi];
        printf("────────────────────────────────────────────\n");
        printf("Prompt #%d: \"%s\"\n", pi + 1, prompt);

        // ── Tokenize ──
        int np = wubu_tokenizer_encode(&tok, prompt, pids, MAX_TOKENS);
        if (np <= 0) {
            printf("  [SKIP] tokenization failed\n\n");
            continue;
        }
        printf("  Tokens: %d\n", np);

        // ── Lookup embeddings ──
        for (int i = 0; i < np; i++) {
            int id = pids[i];
            if (id < 0 || id >= vocab_size) id = 0;
            memcpy(x + i * D, embd + id * D, D * sizeof(float));
        }

        // ── OUR ENGINE: forward pass ──
        wubu_model_forward_from_embd(&model, x, 1, np, logits);

        // Extract top-5 from last token position
        float *last_logits = logits + (np - 1) * vocab_size;
        int our_top_ids[TOP_K];
        float our_top_vals[TOP_K];
        find_top_k(last_logits, vocab_size, TOP_K, our_top_ids, our_top_vals);

        // Decode our top-3 tokens
        printf("  Our top-3:\n");
        for (int k = 0; k < 3 && k < TOP_K; k++) {
            char buf[256];
            wubu_tokenizer_decode(&tok, &our_top_ids[k], 1, buf, 255);
            scrub(buf);
            printf("    [%d] '%s' (%.4f)\n", our_top_ids[k], buf, our_top_vals[k]);
        }

        // ── LLAMA.CPP REFERENCE ──
        ref_output[0] = '\0';
        int ref_len = run_llamacpp(prompt, ref_output, MAX_OUTPUT);

        int ref_token_id = -1;
        char ref_token_text[256] = "(failed)";
        if (ref_len > 0) {
            // Try to decode the reference output as a token
            // llama-cli with -n 1 will output the single token's text
            // We use the tokenizer to see if we can find a matching token
            // (llama-cli output is the raw generated text)
            int ref_pids[16];
            int ref_np = wubu_tokenizer_encode(&tok, ref_output, ref_pids, 16);
            if (ref_np > 0) {
                ref_token_id = ref_pids[0];
                char buf2[256];
                wubu_tokenizer_decode(&tok, &ref_token_id, 1, buf2, 255);
                scrub(buf2);
                snprintf(ref_token_text, sizeof(ref_token_text), "%s", buf2);
            } else {
                // Just show raw output
                scrub(ref_output);
                snprintf(ref_token_text, sizeof(ref_token_text), "raw:'%s'", ref_output);
            }
        }

        printf("  llama.cpp top-1: '%s'", ref_token_text);
        if (ref_token_id >= 0) printf(" (token %d)", ref_token_id);
        printf("\n");

        // ── COMPARISON ──
        int match = (ref_token_id >= 0 && our_top_ids[0] == ref_token_id);
        if (match) {
            printf("  >>> TOP-1 MATCH!\n");
            total_matches++;
        } else {
            printf("  >>> TOP-1 MISMATCH (our=%d, ref=%d)\n",
                   our_top_ids[0], ref_token_id);
        }

        // Jaccard: our top-5 vs {ref} + top-4 from our set (since ref set is incomplete)
        // We report our top-5 self-Jaccard as baseline, and note if ref is in our top-5
        int ref_present = (ref_token_id >= 0) && in_set(ref_token_id, our_top_ids, TOP_K);
        printf("  Ref in our top-5: %s\n", ref_present ? "YES" : "NO");
        
        // If ref matches, compute simulated reference set: {ref} + next 4 from our
        if (ref_token_id >= 0) {
            int ref_set[TOP_K];
            ref_set[0] = ref_token_id;
            // Fill rest with our next-best unique ids
            int filled = 1;
            for (int k = 0; k < TOP_K && filled < TOP_K; k++) {
                if (our_top_ids[k] != ref_token_id) {
                    ref_set[filled++] = our_top_ids[k];
                }
            }
            while (filled < TOP_K) ref_set[filled++] = 0;
            float j = jaccard(our_top_ids, ref_set, TOP_K);
            printf("  Jaccard(our_top5, ref_simulated_top5): %.3f\n", j);
            total_jaccard += j;
        }

        total_prompts++;
        printf("\n");
    }

    // ── SUMMARY ──
    printf("════════════════════════════════════════\n");
    printf("SUMMARY:\n");
    printf("  Total prompts: %d\n", total_prompts);
    printf("  Top-1 matches: %d/%d (%.1f%%)\n",
           total_matches, total_prompts,
           100.0f * total_matches / total_prompts);
    if (total_prompts > 0) {
        printf("  Avg Jaccard (simulated): %.3f\n",
               total_jaccard / total_prompts);
    }
    printf("════════════════════════════════════════\n");

    // ── Cleanup ──
    free(embd);
    free(pids);
    free(x);
    free(logits);
    wubu_tokenizer_free(&tok);
    wubu_model_free(&model);

    printf("=== REGRESSION TEST COMPLETE ===\n");
    return 0;
}
