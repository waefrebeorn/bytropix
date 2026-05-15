#include "wubu_tokenizer.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *prompt = "Hello, what is quantum computing?";
    
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, path)) {
        fprintf(stderr, "Failed to init tokenizer\n");
        return 1;
    }
    
    const int IM_START = 248045;
    const int IM_END   = 248046;
    const int THINK    = 248068;
    const int NL_TOKEN = 198;
    
    int pids[65536];
    int pos = 0;
    
    // BOS
    pids[pos++] = tok.bos_id;
    
    // <|im_start|>system\nYou are a helpful assistant.
    pids[pos++] = IM_START;
    {   int n = wubu_tokenizer_encode(&tok, "system\nYou are a helpful assistant.",
                                      pids + pos, 65535 - pos);
        printf("System msg: %d tokens\n", n);
        pos += n;
    }
    
    // <|im_end|>\n<|im_start|>user\n
    pids[pos++] = IM_END;
    pids[pos++] = NL_TOKEN;
    pids[pos++] = IM_START;
    {   int n = wubu_tokenizer_encode(&tok, "user\n",
                                      pids + pos, 65535 - pos);
        printf("User role: %d tokens\n", n);
        pos += n;
    }
    
    // [USER_PROMPT]
    {   int n = wubu_tokenizer_encode(&tok, prompt,
                                      pids + pos, 65535 - pos);
        printf("User prompt: %d tokens\n", n);
        pos += n;
    }
    
    // <|im_end|>\n<|im_start|>assistant\n<think>\n
    pids[pos++] = IM_END;
    pids[pos++] = NL_TOKEN;
    pids[pos++] = IM_START;
    {   int n = wubu_tokenizer_encode(&tok, "assistant\n",
                                      pids + pos, 65535 - pos);
        printf("Assistant prefix: %d tokens\n", n);
        pos += n;
    }
    pids[pos++] = THINK;
    pids[pos++] = NL_TOKEN;
    
    int np = pos;
    printf("\nTotal chat tokens: %d\n", np);
    
    printf("\n=== Token dump (first 35) ===\n");
    for (int i = 0; i < np && i < 35; i++) {
        char buf[256] = {0};
        wubu_tokenizer_decode(&tok, pids + i, 1, buf, 255);
        const char *marker = "";
        if (pids[i] == IM_START) marker = " <|im_start|>";
        else if (pids[i] == IM_END) marker = " <|im_end|>";
        else if (pids[i] == THINK) marker = " <think>";
        else if (pids[i] == tok.bos_id) marker = " <BOS>";
        printf("  [%d] id=%d '%s'%s\n", i, pids[i], buf, marker);
    }
    if (np > 35) {
        printf("  ... (skipping %d tokens) ...\n", np - 35 - 5);
        for (int i = np - 5; i < np; i++) {
            char buf[256] = {0};
            wubu_tokenizer_decode(&tok, pids + i, 1, buf, 255);
            const char *marker = "";
            if (pids[i] == IM_START) marker = " <|im_start|>";
            else if (pids[i] == IM_END) marker = " <|im_end|>";
            else if (pids[i] == THINK) marker = " <think>";
            printf("  [%d] id=%d '%s'%s\n", i, pids[i], buf, marker);
        }
    }
    
    // Full decode
    char decoded[65536];
    int nd = wubu_tokenizer_decode(&tok, pids, np, decoded, 65535);
    if (nd > 0) {
        decoded[nd] = '\0';
        printf("\n=== Full decoded output ===\n%s\n", decoded);
    }
    
    // Check special tokens
    printf("\n=== Special token presence ===\n");
    int found_start = 0, found_end = 0, found_think = 0;
    for (int i = 0; i < np; i++) {
        if (pids[i] == IM_START) found_start++;
        if (pids[i] == IM_END) found_end++;
        if (pids[i] == THINK) found_think++;
    }
    printf("  <|im_start|>: %d occurrences\n", found_start);
    printf("  <|im_end|>:   %d occurrences\n", found_end);
    printf("  <think>:      %d occurrences (%s)\n", found_think, found_think >= 1 ? "OK" : "MISSING");
    
    // Verify order
    printf("\n=== Structure verification ===\n");
    printf("  BOS present: %s\n", pids[0] == tok.bos_id ? "YES" : "NO");
    printf("  Starts with im_start: %s\n", pids[1] == IM_START ? "YES" : "NO");
    printf("  System+im_end present: ");
    int ok = 0;
    for (int i = 2; i < np; i++) {
        if (pids[i] == IM_END) { ok = 1; break; }
    }
    printf("%s\n", ok ? "YES" : "NO");
    printf("  Ends with think+nl: %s\n", (np >= 2 && pids[np-2] == THINK && pids[np-1] == NL_TOKEN) ? "YES" : "NO");
    
    wubu_tokenizer_free(&tok);
    return 0;
}
