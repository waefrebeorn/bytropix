#include "wubu_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <gguf_path> [text]\n", argv[0]);
        printf("  If text is given, it is tokenized and the result printed.\n");
        printf("  If no text, runs built-in tests.\n");
        return 1;
    }
    
    const char *model_path = argv[1];
    
    printf("Loading tokenizer from %s...\n", model_path);
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, model_path)) {
        fprintf(stderr, "Failed to load tokenizer\n");
        return 1;
    }
    printf("  Vocab: %d tokens\n", tok.vocab_size);
    printf("  BOS: %d, EOS: %d, PAD: %d\n", tok.bos_id, tok.eos_id, tok.pad_id);
    printf("  Merges: %d\n", tok.n_merges);
    
    if (argc > 2) {
        // Tokenize the provided text
        const char *text = argv[2];
        printf("\nEncoding: \"%s\"\n", text);
        
        int ids[65536];
        int n = wubu_tokenizer_encode(&tok, text, ids, 65536);
        printf("  Token IDs (%d): ", n);
        for (int i = 0; i < n && i < 20; i++) printf("%d ", ids[i]);
        if (n > 20) printf("... ");
        printf("\n");
        
        // Decode back
        char decoded[65536];
        int nd = wubu_tokenizer_decode(&tok, ids, n, decoded, 65536);
        printf("  Decoded: \"%s\"\n", decoded);
        
        // Compare with Python
        printf("\nUsing Python tokenizer...\n");
        int py_ids[65536];
        int pyn = wubu_tokenizer_encode_python(&tok, text, py_ids, 65536);
        if (pyn > 0) {
            printf("  Python IDs (%d): ", pyn);
            for (int i = 0; i < pyn && i < 20; i++) printf("%d ", py_ids[i]);
            if (pyn > 20) printf("... ");
            printf("\n");
            
            // Compare
            int match = 1;
            if (n != pyn) {
                printf("  Mismatch: C=%d ids, Python=%d ids\n", n, pyn);
                match = 0;
            } else {
                for (int i = 0; i < n; i++) {
                    if (ids[i] != py_ids[i]) {
                        printf("  First diff at position %d: C=%d Python=%d\n", i, ids[i], py_ids[i]);
                        match = 0;
                        break;
                    }
                }
            }
            if (match) printf("  C tokenization matches Python! ✓\n");
        } else {
            printf("  Python failed\n");
        }
    } else {
        // Built-in tests
        const char *tests[] = {
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "12345",
            "Qwen3.6-35B-A3B",
            "I'm testing contractions like don't and can't.",
            "你好世界",  // Chinese
            "日本語",    // Japanese  
            NULL
        };
        
        printf("\n=== Built-in Tests ===\n\n");
        for (int ti = 0; tests[ti]; ti++) {
            printf("Input: \"%s\"\n", tests[ti]);
            int ids[256];
            int n = wubu_tokenizer_encode(&tok, tests[ti], ids, 256);
            printf("  Tokens (%d): ", n);
            for (int i = 0; i < n && i < 15; i++) printf("%d ", ids[i]);
            printf("\n");
            
            // Compare with Python
            int py_ids[256];
            int pyn = wubu_tokenizer_encode_python(&tok, tests[ti], py_ids, 256);
            printf("  Python (%d): ", pyn);
            for (int i = 0; i < pyn && i < 15; i++) printf("%d ", py_ids[i]);
            printf("\n");
            
            int match = (n == pyn);
            if (match) {
                for (int i = 0; i < n && i < pyn; i++) {
                    if (ids[i] != py_ids[i]) { match = 0; break; }
                }
            }
            printf("  %s\n", match ? "✓ Match" : "✗ Mismatch");
            printf("\n");
        }
    }
    
    wubu_tokenizer_free(&tok);
    return 0;
}
