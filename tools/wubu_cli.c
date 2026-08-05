/*
 * wubu_cli.c -- WuBu-35M operational runner (the mustard seed CLI).
 *
 * Loads the real released checkpoint + the byte-level BPE tokenizer,
 * and generates text. This is the base model that grows: the AGI
 * brain-cluster loop trains it further, adds parameters, and feeds it
 * everything the research repos learn.
 *
 * Usage: wubu_cli [--model models/wubu/model.safetensors]
 *                  [--tok models/wubu/tokenizer.json]
 *                  [--prompt "text"] [--tokens N] [--temp 0.8] [--seed N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu.h"
#include "wubu_tokenizer_hf.h"
#include "wubu_banner.h"

static const char *arg_get(int argc, char **argv, const char *name,
                           const char *def)
{
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    return def;
}

static int arg_int(int argc, char **argv, const char *name, int def)
{
    const char *v = arg_get(argc, argv, name, NULL);
    return v ? atoi(v) : def;
}

int main(int argc, char **argv)
{
    wubu_print_banner("CLI Runner — WuBu-35M", "Base model · byte-level BPE");

    const char *model_path = arg_get(argc, argv, "--model",
                                     "models/wubu/model.safetensors");
    const char *tok_path = arg_get(argc, argv, "--tok",
                                   "models/wubu/tokenizer.json");
    const char *prompt = arg_get(argc, argv, "--prompt",
                                 "The future of efficient language models is");
    int max_new = arg_int(argc, argv, "--tokens", 48);
    float temp = (float)atof(arg_get(argc, argv, "--temp", "0.8"));
    uint32_t seed = (uint32_t)arg_int(argc, argv, "--seed", 42);

    printf("wubu: loading %s ...\n", model_path);
    wubu_model_t m;
    if (wubu_load(&m, model_path) != 0) {
        fprintf(stderr, "wubu: failed to load the model\n");
        return 1;
    }
    wubu_print_stat("Parameters", "%ld (release: %d)",
                    wubu_parameter_count(&m), WUBU_PARAMS);

    printf("wubu: loading %s ...\n", tok_path);
    wubu_tok_hf_t *tok = wubu_tok_hf_load(tok_path);
    if (!tok) {
        fprintf(stderr, "wubu: failed to load the tokenizer\n");
        wubu_free(&m, NULL);
        return 1;
    }
    wubu_print_stat("Vocab", "%d", wubu_tok_hf_vocab_size(tok));

    int *ids = (int *)malloc(sizeof(int) * (WUBU_MAX_SEQ));
    int n_prompt = wubu_tok_hf_encode(tok, prompt, ids, WUBU_MAX_SEQ - 16);
    if (n_prompt <= 0) {
        fprintf(stderr, "wubu: prompt encoded to 0 tokens\n");
        wubu_free(&m, NULL); wubu_tok_hf_free(tok); free(ids);
        return 1;
    }
    printf("wubu: prompt = %d tokens\n", n_prompt);

    wubu_buf_t b;
    if (wubu_buf_alloc(&b, WUBU_MAX_SEQ) != 0) {
        fprintf(stderr, "wubu: failed to allocate the buffer\n");
        wubu_free(&m, NULL); wubu_tok_hf_free(tok); free(ids);
        return 1;
    }

    /* the prompt (uint16 ids) */
    uint16_t *gen = (uint16_t *)malloc(sizeof(uint16_t) * WUBU_MAX_SEQ);
    for (int i = 0; i < n_prompt; i++) gen[i] = (uint16_t)ids[i];

    printf("wubu: generating %d tokens (temp %.1f, seed %u) ...\n",
           max_new, temp, seed);
    size_t made = wubu_generate(&m, &b, gen, (size_t)n_prompt,
                                (size_t)max_new, temp, seed);

    /* decode the full sequence */
    int *all = (int *)malloc(sizeof(int) * (n_prompt + (int)made));
    for (int i = 0; i < n_prompt + (int)made; i++) all[i] = gen[i];
    char *text = wubu_tok_hf_decode(tok, all, n_prompt + (int)made);
    if (text) {
        printf("\n");
        wubu_print_section("Output");
        printf("%s\n", text);
        free(text);
    } else {
        printf("wubu: (decode failed -- tokens %d)\n", n_prompt + (int)made);
        for (int i = 0; i < (int)made; i++) printf("%u ", gen[n_prompt + i]);
        printf("\n");
    }

    free(all); free(gen); free(ids);
    wubu_free(&m, &b);
    wubu_tok_hf_free(tok);
    printf("wubu: done\n");
    return 0;
}
