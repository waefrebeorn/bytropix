#include "wubu_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    printf("START\n"); fflush(stdout);
    
    const char *model_path = "/mnt/wslg/distro/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    if (argc > 1) model_path = argv[1];
    
    printf("GGUF: %s\n", model_path); fflush(stdout);
    
    wubu_tokenizer_t tok;
    memset(&tok, 0, sizeof(tok));
    
    printf("Calling init...\n"); fflush(stdout);
    int ok = wubu_tokenizer_init(&tok, model_path);
    printf("Init returned %d\n", ok); fflush(stdout);
    
    if (!ok) {
        printf("FAIL\n");
        return 1;
    }
    
    printf("PASS: vocab=%d merges=%d\n", tok.vocab_size, tok.n_merges);
    
    const char *text = argc > 2 ? argv[2] : "Hello, world!";
    printf("Encoding: %s\n", text); fflush(stdout);
    
    int ids[1024];
    int n = wubu_tokenizer_encode(&tok, text, ids, 1024);
    printf("Tokens: %d\n", n);
    for (int i = 0; i < n && i < 20; i++) printf("  %d ", ids[i]);
    printf("\n");
    
    wubu_tokenizer_free(&tok);
    return 0;
}
