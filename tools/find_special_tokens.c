/*
 * find_special_tokens.c — Find <|im_start|>, <|im_end|>, <|think|> token IDs
 * by scanning the tokenizer vocabulary.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gguf_reader.h"
#include "wubu_tokenizer.h"

int main() {
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, path)) {
        fprintf(stderr, "FAIL: tokenizer\n"); return 1;
    }
    
    // Search for special tokens in vocabulary
    const char *targets[] = {"<|im_start|>", "<|im_end|>", "<|endoftext|>", 
                             "<|think|>", "<|assistant|>", "<|user|>",
                             "<|system|>", "<|im_start|", "<|im_end|",
                             "<s>", "</s>", NULL};
    
    for (int t = 0; targets[t]; t++) {
        int found = -1;
        for (int i = 0; i < tok.vocab_size; i++) {
            if (strcmp(tok.vocab[i].bytes, targets[t]) == 0) {
                found = i;
                break;
            }
        }
        if (found >= 0) {
            printf("  '%s' => ID %d\n", targets[t], found);
        } else {
            // Try prefix match
            for (int i = 0; i < tok.vocab_size && i < 250000; i++) {
                if (strstr(tok.vocab[i].bytes, targets[t])) {
                    printf("  '%s' => FOUND PARTIAL: '%.*s' = ID %d\n", targets[t], tok.vocab[i].byte_len, tok.vocab[i].bytes, i);
                    break;
                }
            }
        }
    }
    
    // Print special IDs from GGUF KV
    printf("\nGGUF special IDs:\n");
    printf("  BOS=%d, EOS=%d, PAD=%d\n", tok.bos_id, tok.eos_id, tok.pad_id);
    
    // Check if these token IDs exist and what string they map to
    int check_ids[] = {248044, 248046, 248053, 248054, 248055, 248056, 248057, 0, 1, 2};
    for (int i = 0; i < 10; i++) {
        int id = check_ids[i];
        if (id >= 0 && id < tok.vocab_size) {
            printf("  ID %d => '%.*s'\n", id, tok.vocab[id].byte_len, tok.vocab[id].bytes);
        }
    }
    
    wubu_tokenizer_free(&tok);
    return 0;
}
