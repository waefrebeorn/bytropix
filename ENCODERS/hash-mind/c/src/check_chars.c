#include "tokenizer.h"
#include <stdio.h>

int main() {
    Tokenizer tok;
    tokenizer_init(&tok);
    
    /* Check what codes each ASCII char maps to */
    unsigned char chars_seen[256] = {0};
    
    FILE* f = fopen("training_sample.txt", "rb");
    char buf[50002];
    int n = fread(buf, 1, 50000, f);
    buf[n] = 0;
    fclose(f);
    
    for (int i = 0; i < n; i++) {
        unsigned char c = (unsigned char)buf[i];
        chars_seen[c] = 1;
    }
    
    printf("Characters in CORPUS sample (256 total):\n");
    int unk_count = 0;
    for (int c = 0; c < 256; c++) {
        if (chars_seen[c]) {
            int idx = tok.char_to_idx[c];
            if (idx == 0 && c != '0') {
                printf("  %3d (0x%02x) '%c' -> UNMAPPED (default 0)\n", c, c, c > 31 && c < 127 ? c : '.');
                unk_count++;
            } else {
                printf("  %3d (0x%02x) '%c' -> %d\n", c, c, c > 31 && c < 127 ? c : '.', idx);
            }
        }
    }
    printf("\nTotal unmapped chars in CORPUS: %d\n", unk_count);
    return 0;
}
