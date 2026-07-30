#include "wubu_tokenizer_hf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path = "/home/wubu/models/Agents-A1-4B/tokenizer.json";
    if (argc > 1) path = argv[1];
    wubu_tok_hf_t *t = wubu_tok_hf_load(path);
    if (!t) { fprintf(stderr, "load failed\n"); return 1; }
    fprintf(stderr, "load ok bos=%d eos=%d vocab=%d\n",
            wubu_tok_hf_bos_id(t), wubu_tok_hf_eos_id(t), wubu_tok_hf_vocab_size(t));
    const char *samples[] = {"Hello", "hello", "Hello world", NULL};
    for (int i = 0; samples[i]; i++) {
        int ids[64]; int n = wubu_tok_hf_encode(t, samples[i], ids, 64);
        char *dec = wubu_tok_hf_decode(t, ids, n);
        fprintf(stderr, "in='%s' n=%d first=%d dec='%s'\n",
                samples[i], n, n > 0 ? ids[0] : -1, dec ? dec : "(null)");
        free(dec);
    }
    wubu_tok_hf_free(t);
    return 0;
}
