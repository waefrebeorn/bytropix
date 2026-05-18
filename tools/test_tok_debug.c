// Debug the tokenizer multi-token crash
#include "wubu_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) {
        fprintf(stderr, "Failed to init tokenizer\n");
        return 1;
    }
    printf("Tokenizer initialized OK\n");
    printf("vocab_size=%d bos=%d eos=%d\n", tok.vocab_size, tok.bos_id, tok.eos_id);
    printf("byte_token_ids[0]=%d byte_token_ids[65]=%d\n", 
           tok.byte_token_ids[0], tok.byte_token_ids[65]);

    const char *test_inputs[] = {
        "a",
        "ab", 
        "hello",
        "hello world",
        "The capital of France is",
        NULL
    };

    int out[128];
    for (int t = 0; test_inputs[t]; t++) {
        printf("\n--- Encode '%s' ---\n", test_inputs[t]);
        int n = wubu_tokenizer_encode(&tok, test_inputs[t], out, 128);
        printf("Returned: %d tokens\n", n);
        if (n > 0) {
            printf("Tokens:");
            for (int i = 0; i < n && i < 20; i++) printf(" %d", out[i]);
            printf("\n");
            // Decode back
            char buf[1024];
            int nc = wubu_tokenizer_decode(&tok, out, n, buf, 1024);
            printf("Decoded: '");
            if (nc > 0) fwrite(buf, 1, nc, stdout);
            printf("'\n");
        }
    }

    wubu_tokenizer_free(&tok);
    return 0;
}
