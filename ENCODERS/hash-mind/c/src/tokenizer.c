#include "tokenizer.h"

void tokenizer_init(Tokenizer* tok) {
    memset(tok->char_to_idx, 0, sizeof(tok->char_to_idx));
    memset(tok->idx_to_char, 0, sizeof(tok->idx_to_char));

    /* Maps all printable ASCII (32-126) + \n(10) + \t(9) */
    int idx = 0;
    
    /* Control chars first */
    tok->char_to_idx['\t'] = idx;
    tok->idx_to_char[idx] = '\t';
    idx++;
    
    tok->char_to_idx['\n'] = idx;
    tok->idx_to_char[idx] = '\n';
    idx++;
    
    /* Space */
    tok->char_to_idx[' '] = idx;
    tok->idx_to_char[idx] = ' ';
    idx++;
    
    /* Printable ASCII: 33 ('!') to 126 ('~') */
    for (char c = 33; c <= 126; c++) {
        tok->char_to_idx[(unsigned char)c] = idx;
        tok->idx_to_char[idx] = c;
        idx++;
    }
    
    tok->vocab_size = idx;
}

int tokenizer_encode(const Tokenizer* tok, const char* text, int* output, int max_len) {
    int len = 0;
    for (int i = 0; text[i] && len < max_len; i++) {
        unsigned char uc = (unsigned char)text[i];
        int idx = tok->char_to_idx[uc];
        /* Only map known chars — skip non-ASCII / unmapped */
        if (idx != 0 || uc == 0 || uc == tok->idx_to_char[0]) {
            output[len++] = idx;
        }
        /* else skip the character entirely */
    }
    return len;
}

int tokenizer_decode(const Tokenizer* tok, const int* tokens, int n,
                      char* output, int max_len) {
    int pos = 0;
    for (int i = 0; i < n && pos < max_len - 1; i++) {
        int tid = tokens[i];
        if (tid >= 0 && tid < tok->vocab_size) {
            output[pos++] = tok->idx_to_char[tid];
        } else {
            output[pos++] = '?';
        }
    }
    output[pos] = '\0';
    return pos;
}
