#include "hashmind_data.h"
#include "hashmind_model.h"
#include "rolling_hash.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void textdata_init(TextData* td, Tokenizer* tok, const char* text) {
    td->tok = tok;
    int raw_len = (int)strlen(text);
    td->tokens = (int*)malloc(raw_len * sizeof(int));
    int n = tokenizer_encode(tok, text, td->tokens, raw_len);
    td->num_tokens = n;
    td->hashes = (uint32_t*)malloc((n - HASH_WINDOW + 1) * sizeof(uint32_t));
    int nh;
    rolling_hash_all(td->tokens, n, HASH_WINDOW, td->hashes, &nh);
    td->cursor = 0;
    td->epoch = 0;

    /* We need n - HASH_WINDOW + 1 training examples (each predicts next char) */
    /* The "valid" range for cursor: 0 to (n - HASH_WINDOW - CONTEXT_LEN) */
    /* We store hashes starting from index HASH_WINDOW-1 */
}

void textdata_free(TextData* td) {
    free(td->tokens);
    free(td->hashes);
}

void textdata_reset(TextData* td) {
    td->cursor = 0;
    td->epoch++;
}

int textdata_next(TextData* td, TrainExample* ex) {
    int max_start = td->num_tokens - CONTEXT_LEN - 1;
    if (max_start < 0) return 0;

    if (td->cursor >= max_start) {
        td->cursor = 0;
        td->epoch++;
    }

    int start = td->cursor;
    td->cursor++;

    /* Input tokens: CONTEXT_LEN tokens ending at start */
    for (int i = 0; i < CONTEXT_LEN; i++) {
        int idx = start - CONTEXT_LEN + 1 + i;
        if (idx < 0) idx = 0;
        ex->input_tokens[i] = td->tokens[idx];

        /* Hash for this position: hash of window ending at idx */
        int hash_idx = idx - HASH_WINDOW + 1;
        if (hash_idx < 0) hash_idx = 0;
        ex->input_hashes[i] = td->hashes[hash_idx];
    }

    /* Target: next token */
    ex->target_token = td->tokens[start + 1];

    return 1;
}

float textdata_eval(TextData* td, Tokenizer* tok, const char* test_text,
                    HashMindModel* model) {
    int tokens[512];
    int n = tokenizer_encode(tok, test_text, tokens, 512);
    if (n < CONTEXT_LEN + 1) return -1;

    uint32_t hashes[512];
    int nh;
    rolling_hash_all(tokens, n, HASH_WINDOW, hashes, &nh);

    float total_loss = 0;
    int count = 0;

    for (int i = CONTEXT_LEN; i < n - 1; i++) {
        float logits[VOCAB_SIZE];
        BlockActs acts;
        hashmind_forward(model, hashes + i - CONTEXT_LEN, CONTEXT_LEN,
                         tokens + i - CONTEXT_LEN, CONTEXT_LEN, logits, &acts);
        float loss = nn_cross_entropy_loss(logits, tokens[i + 1], VOCAB_SIZE);
        total_loss += loss;
        count++;
    }

    return count > 0 ? total_loss / count : -1;
}

void textdata_generate_sample(HashMindModel* model, Tokenizer* tok,
                               const char* prompt, int max_new_tokens,
                               float temperature, char* output_buf, int buf_size) {
    int indices[MAX_SEQ];
    int len = tokenizer_encode(tok, prompt, indices, MAX_SEQ);
    int pos = 0;

    /* Copy prompt to output */
    int plen = (int)strlen(prompt);
    if (plen < buf_size) {
        memcpy(output_buf, prompt, plen);
        pos = plen;
    }

    srand((unsigned int)time(NULL));
    uint32_t hashes[MAX_SEQ];
    int nh;

    for (int step = 0; step < max_new_tokens && pos < buf_size - 1; step++) {
        rolling_hash_all(indices, len, HASH_WINDOW, hashes, &nh);
        int next = hashmind_generate(model, indices, len, hashes, temperature);
        indices[len++] = next;
        if (len >= MAX_SEQ) break;
        output_buf[pos++] = tok->idx_to_char[next];
    }
    output_buf[pos] = '\0';
}
