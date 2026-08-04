/*
 * tok_test.c — encode a prompt with wubu_tokenizer_hf and dump token ids.
 */
#include "wubu_tokenizer_hf.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <tokenizer.json> <text>\n", argv[0]); return 2; }
    wubu_tok_hf_t *t = wubu_tok_hf_load(argv[1]);
    if (!t) { fprintf(stderr, "load failed\n"); return 1; }
    printf("vocab_size=%d\n", wubu_tok_hf_vocab_size(t));
    int ids[512];
    int n = wubu_tok_hf_encode(t, argv[2], ids, 512);
    printf("rc=%d\n", n);
    for (int i = 0; i < n; i++) printf("  [%d] %d\n", i, ids[i]);
    wubu_tok_hf_free(t);
    return 0;
}
