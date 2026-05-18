#include <stdio.h>
#include <string.h>
#include "gguf_reader.h"
#include "wubu_tokenizer.h"

int main() {
    wubu_tokenizer_t tok;
    wubu_tokenizer_init(&tok, "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    
    const char *targets[] = {"<|think|>", "<think>", "think", NULL};
    for (int t = 0; targets[t]; t++) {
        for (int i = 0; i < tok.vocab_size; i++) {
            if (strcmp(tok.vocab[i].bytes, targets[t]) == 0) {
                printf("'%s' => ID %d\n", targets[t], i);
                break;
            }
        }
    }
    
    // Encode "<think>"
    int ids[10];
    int n = wubu_tokenizer_encode(&tok, "<think>", ids, 10);
    printf("encode '<think>': %d tokens:", n);
    for (int i = 0; i < n; i++) printf(" %d", ids[i]);
    printf("\n");
    
    // Encode "\\n" 
    n = wubu_tokenizer_encode(&tok, "\n", ids, 10);
    printf("encode '\\\\n': %d tokens:", n);
    for (int i = 0; i < n; i++) printf(" %d", ids[i]);
    printf("\n");
    
    // Encode full assistant prefix
    n = wubu_tokenizer_encode(&tok, "assistant\n", ids, 10);
    printf("encode 'assistant\\\\n': %d tokens:", n);
    for (int i = 0; i < n; i++) printf(" %d", ids[i]);
    printf("\n");
    
    wubu_tokenizer_free(&tok);
    return 0;
}
