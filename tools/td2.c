#include "wubu_tokenizer.h"
#include <stdio.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    fprintf(stderr, "sizeof(tok) = %zu\n", sizeof(wubu_tokenizer_t));
    fprintf(stderr, "sizeof(wubu_token_t) = %zu\n", sizeof(wubu_token_t));
    fprintf(stderr, "MAX_VOCAB = %d, alloc = %zu\n", WUBU_TOKENIZER_MAX_VOCAB, WUBU_TOKENIZER_MAX_VOCAB * sizeof(wubu_token_t));
    fprintf(stderr, "MAX_MERGES = %d, alloc = %zu\n", WUBU_TOKENIZER_MAX_MERGES, WUBU_TOKENIZER_MAX_MERGES * sizeof(wubu_merge_t));
    
    wubu_tokenizer_t tok;
    fprintf(stderr, "Tok on stack at %p\n", &tok);
    fflush(stderr);
    
    bool ok = wubu_tokenizer_init(&tok, path);
    fprintf(stderr, "Init: %d\n", ok);
    
    if (ok) {
        fprintf(stderr, "vocab=%d merges=%d\n", tok.vocab_size, tok.n_merges);
        wubu_tokenizer_free(&tok);
    }
    return ok ? 0 : 1;
}
