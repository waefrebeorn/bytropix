#include "wubu_tokenizer.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    fprintf(stderr, "Starting tokenizer test...\n");
    
    wubu_tokenizer_t tok;
    
    if (!wubu_tokenizer_init_from_files(&tok,
            "data/tokenizer_vocab.txt",
            "data/tokenizer_merges.txt",
            -1, 248046, 248044)) {
        fprintf(stderr, "Failed to init tokenizer\n");
        return 1;
    }
    
    fprintf(stderr, "Tokenizer: %d tokens, %d merges\n", tok.vocab_size, tok.n_merges);
    
    const char *tests[] = {"Hello, world!", "Qwen3.6", "test123", "你好", NULL};
    for (int i = 0; tests[i]; i++) {
        int ids[256];
        int n = wubu_tokenizer_encode(&tok, tests[i], ids, 256);
        fprintf(stderr, "\"%s\" -> %d tokens: ", tests[i], n);
        for (int j = 0; j < n && j < 10; j++) fprintf(stderr, "%d ", ids[j]);
        fprintf(stderr, "\n");
    }
    
    wubu_tokenizer_free(&tok);
    return 0;
}
