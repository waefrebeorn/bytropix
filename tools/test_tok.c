#include "wubu_tokenizer.h"
#include <stdio.h>

int main() {
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"))
        return 1;
    
    int ids[128];
    int n = wubu_tokenizer_encode(&tok, "The meaning of life is to", ids, 128);
    printf("Prompt: %d tokens\n", n);
    for (int i = 0; i < n; i++) {
        char buf[256] = {0};
        wubu_tokenizer_decode(&tok, ids + i, 1, buf, 255);
        printf("  [%d] token %d: '%s'\n", i, ids[i], buf);
    }
    
    // Also check "Default"
    n = wubu_tokenizer_encode(&tok, "Default", ids, 128);
    printf("\n'Default': %d tokens\n", n);
    for (int i = 0; i < n; i++) {
        char buf[256] = {0};
        wubu_tokenizer_decode(&tok, ids + i, 1, buf, 255);
        printf("  token %d: '%s'\n", ids[i], buf);
    }
    
    wubu_tokenizer_free(&tok);
    return 0;
}
