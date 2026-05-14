/**
 * hashmind_data.h — Data Loader for HashMind Training
 *
 * Generates character-level training sequences from text.
 * Supports CORPUS-like data, plain text files, and synthetic patterns.
 */
#ifndef HASHMIND_DATA_H
#define HASHMIND_DATA_H

#include "tokenizer.h"
#include "hashmind_model.h"
#include "rolling_hash.h"

/* ─── Training Example ─── */
typedef struct {
    int input_tokens[CONTEXT_LEN];     /* input context */
    uint32_t input_hashes[CONTEXT_LEN]; /* rolling hashes */
    int target_token;                   /* next token to predict */
} TrainExample;

/* ─── Text Buffer ─── */
typedef struct {
    Tokenizer* tok;
    int* tokens;         /* tokenized text */
    uint32_t* hashes;    /* precomputed rolling hashes */
    int num_tokens;
    int cursor;          /* current position for iteration */
    int epoch;
} TextData;

/* Initialize from raw text */
void textdata_init(TextData* td, Tokenizer* tok, const char* text);

/* Free memory */
void textdata_free(TextData* td);

/* Get next training example. Returns 1 on success, 0 at end of epoch (auto-rewinds). */
int textdata_next(TextData* td, TrainExample* ex);

/* Reset cursor to beginning */
void textdata_reset(TextData* td);

/* Get loss on held-out test sequence */
float textdata_eval(TextData* td, Tokenizer* tok, const char* test_text,
                    HashMindModel* model);

/* Print generation sample */
void textdata_generate_sample(HashMindModel* model, Tokenizer* tok,
                               const char* prompt, int max_new_tokens,
                               float temperature, char* output_buf, int buf_size);

#endif /* HASHMIND_DATA_H */
