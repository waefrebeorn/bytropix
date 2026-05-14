/**
 * tokenizer.h — ASCII Tokenizer
 *
 * Maps all printable ASCII (32-126) + common control chars to integer token IDs.
 * Vocab: all 95 printable ASCII + \n + \t = 97 tokens (same as StandardASCIIConverter).
 */
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <string.h>

#define VOCAB_SIZE      97
#define MAX_SEQ         4096

typedef struct {
    int char_to_idx[256];          /* char → token id */
    char idx_to_char[VOCAB_SIZE];  /* token id → char */
    int vocab_size;
} Tokenizer;

void tokenizer_init(Tokenizer* tok);
int  tokenizer_encode(const Tokenizer* tok, const char* text, int* output, int max_len);
int  tokenizer_decode(const Tokenizer* tok, const int* tokens, int n, char* output, int max_len);

#endif /* TOKENIZER_H */
