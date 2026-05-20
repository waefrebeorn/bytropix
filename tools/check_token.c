#include "wubu_tokenizer.h"
#include <stdio.h>

int main() {
    const char *path = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, path)) return 1;
    
    int ids[] = {92941, 92942, 92943, 248044, 248045, 248046};
    for (int i = 0; i < 6; i++) {
        char buf[256];
        int n = wubu_tokenizer_decode(&tok, &ids[i], 1, buf, 256);
        printf("token %d: ", ids[i]);
        if (n > 0) fwrite(buf, 1, n, stdout);
        printf("\n");
    }
    
    wubu_tokenizer_free(&tok);
    return 0;
}
