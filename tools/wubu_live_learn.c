/* wubu_live_learn.c -- the LIVE SPEED LEARNING loop (the kernel's brain-side).
 *
 * The feedback loop (the user's directive: "live speed learning for AGI,
 * the learning loop is more proper LLM corpus and we feedback loop with
 * nvidia cloud keys"):
 *
 *   1. LOAD the WuBu model (the SAME path the trainer uses — wubu_load,
 *      which handles the WuBu architecture; gen_text's generic loader
 *      does NOT).
 *   2. GENERATE a draft from a prompt (wubu_generate, WuBu-native).
 *   3. (feedback) the NVIDIA NIM oracle (tools/nvidia_nim.py score_draft)
 *      scores the draft; the critique is the SFT target.
 *   4. ACCUMULATE (prompt, draft, critique, score) into a JSONL buffer
 *      that the next SFT training round consumes.
 *
 * The kernel (wubuos wubu_agi_kernel) is the supervisor; this is the
 * brain-side worker that feeds it. Usage:
 *
 *   wubu_live_learn <model.safetensors> <tokenizer.json> <prompt.txt>
 *                   [--steps N] [--out buffer.jsonl] [--temp T]
 *
 * Build: gcc -O2 -std=c11 -I include tools/wubu_live_learn.c src/wubu.c \
 *        ... (the trainer's link line; see Makefile wubu_train)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu.h"
#include "wubu_tokenizer_hf.h"

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.safetensors> <tokenizer.json> <prompt.txt> [--steps N] [--temp T]\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *tok_path   = argv[2];
    const char *prompt_path = argv[3];
    int steps = 8;
    float temp = 0.7f;
    for (int i = 4; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--steps")) steps = atoi(argv[i+1]);
        if (!strcmp(argv[i], "--temp"))  temp  = (float)atof(argv[i+1]);
    }

    /* 1. LOAD (the trainer path — WuBu-arch aware) */
    wubu_model_t m;
    if (wubu_load(&m, model_path) != 0) {
        fprintf(stderr, "cannot load %s\n", model_path);
        return 1;
    }
    fprintf(stderr, "live_learn: %ld params, %d layers\n",
            (long)wubu_parameter_count(&m), m.n_layers);

    /* tokenize the prompt with the HF tokenizer (the model's own) */
    /* NOTE: prompt tokenization needs the HF tokenizer object; the trainer
     * uses .tok streams. For the live loop the prompt comes pre-tokenized
     * or via wubu_tokenizer; see the wrapper script for the full path. */

    wubu_buf_t b;
    if (wubu_buf_alloc(&b, WUBU_MAX_SEQ) != 0) return 1;

    /* read the prompt as a token stream (uint16) — simplest robust path:
     * the wrapper tokenizes with wubu_tokenc and feeds the .tok bytes. */
    FILE *f = fopen(prompt_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", prompt_path); return 1; }
    uint16_t tokens[WUBU_MAX_SEQ];
    size_t n_prompt = 0;
    while (n_prompt < WUBU_MAX_SEQ && fread(&tokens[n_prompt], 2, 1, f) == 1)
        n_prompt++;
    fclose(f);
    fprintf(stderr, "live_learn: %zu prompt tokens\n", n_prompt);

    /* 2. GENERATE */
    size_t total = wubu_generate(&m, &b, tokens, n_prompt, (size_t)steps, temp, 48);
    fprintf(stderr, "live_learn: generated %zu tokens total\n", total);

    /* decode the generated tokens via the HF tokenizer (C-side, no python
     * transformers needed) so the oracle can score the actual text */
    {
        wubu_tok_hf_t *hf = wubu_tok_hf_load(tok_path);
        if (hf) {
            int ngen = (int)(total - n_prompt);
            if (ngen > 0) {
                int ids[1024];
                for (int i = 0; i < ngen && i < 1024; i++)
                    ids[i] = tokens[n_prompt + i];
                char *txt = wubu_tok_hf_decode(hf, ids, ngen);
                if (txt) {
                    fprintf(stderr, "live_learn DRAFT: %s\n", txt);
                    free(txt);
                }
            }
        }
    }

    /* 3+4. the NVIDIA oracle + buffer accumulation happen in the WRAPPER
     * (python): it decodes the token stream, calls nvidia_nim.score_draft,
     * and appends {prompt, draft, critique, score} to the SFT buffer.
     * This C binary writes the raw generated token stream to stdout. */
    fwrite(tokens + n_prompt, 2, total - n_prompt, stdout);

    wubu_free(&m, &b);
    return 0;
}
