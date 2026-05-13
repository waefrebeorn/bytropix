/*
 * tokenize_corpus.c — Tokenize corpus_raw.txt using Qwen3.6 tokenizer
 * and write binary uint32_t data for training.
 *
 * Usage: ./tokenize_corpus <gguf_path>
 *   Reads data/corpus_raw.txt, tokenizes in chunks, writes data/train_data.bin
 *   and data/train_meta.txt.
 */
#include "wubu_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define CHUNK_SIZE 60000   /* max input chars per tokenizer call (must be <= 65536) */
#define MAX_TOKENS_PER_CHUNK 131072
#define OUTPUT_PATH "data/train_data.bin"
#define META_PATH   "data/train_meta.txt"

int main(int argc, char **argv) {
    const char *model_path;
    if (argc > 1) {
        model_path = argv[1];
    } else {
        model_path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    }

    fprintf(stderr, "Loading tokenizer from %s...\n", model_path);
    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, model_path)) {
        fprintf(stderr, "FAILED: wubu_tokenizer_init\n");
        return 1;
    }
    fprintf(stderr, "  Vocab: %d tokens\n", tok.vocab_size);
    fprintf(stderr, "  BOS: %d, EOS: %d, PAD: %d\n", tok.bos_id, tok.eos_id, tok.pad_id);
    fprintf(stderr, "  Merges: %d\n", tok.n_merges);

    /* Open corpus file */
    const char *corpus_path = "data/corpus_raw.txt";
    fprintf(stderr, "Reading corpus from %s...\n", corpus_path);
    FILE *fcorpus = fopen(corpus_path, "rb");
    if (!fcorpus) {
        fprintf(stderr, "FAILED: cannot open %s\n", corpus_path);
        wubu_tokenizer_free(&tok);
        return 1;
    }

    /* Get file size */
    fseek(fcorpus, 0, SEEK_END);
    long fsize = ftell(fcorpus);
    fseek(fcorpus, 0, SEEK_SET);
    fprintf(stderr, "  Corpus size: %ld bytes\n", fsize);

    /* Read entire corpus */
    char *corpus = (char *)malloc(fsize + 1);
    if (!corpus) {
        fprintf(stderr, "FAILED: malloc(%ld)\n", fsize + 1);
        fclose(fcorpus);
        wubu_tokenizer_free(&tok);
        return 1;
    }
    size_t bytes_read = fread(corpus, 1, fsize, fcorpus);
    fclose(fcorpus);
    if ((long)bytes_read != fsize) {
        fprintf(stderr, "FAILED: read only %zu of %ld bytes\n", bytes_read, fsize);
        free(corpus);
        wubu_tokenizer_free(&tok);
        return 1;
    }
    corpus[fsize] = '\0';
    fprintf(stderr, "  Read OK\n");

    /* Open output file */
    fprintf(stderr, "Opening output %s...\n", OUTPUT_PATH);
    FILE *fout = fopen(OUTPUT_PATH, "wb");
    if (!fout) {
        fprintf(stderr, "FAILED: cannot open %s\n", OUTPUT_PATH);
        free(corpus);
        wubu_tokenizer_free(&tok);
        return 1;
    }

    /* Tokenize in chunks */
    int *all_tokens = NULL;
    size_t all_tokens_cap = 0;
    size_t total_tokens = 0;
    long pos = 0;

    /* Allocate temp buffers */
    char *chunk = (char *)malloc(CHUNK_SIZE + 4);
    int *tmp_ids = (int *)malloc(MAX_TOKENS_PER_CHUNK * sizeof(int));
    if (!chunk || !tmp_ids) {
        fprintf(stderr, "FAILED: malloc for chunk buffers\n");
        free(chunk); free(tmp_ids); free(corpus); fclose(fout);
        wubu_tokenizer_free(&tok);
        return 1;
    }

    while (pos < fsize) {
        /* Determine chunk size */
        long remaining = fsize - pos;
        long this_chunk = remaining < CHUNK_SIZE ? remaining : CHUNK_SIZE;

        /* Copy to chunk buffer, ensuring null termination */
        memcpy(chunk, corpus + pos, this_chunk);
        chunk[this_chunk] = '\0';

        /* Trim trailing whitespace to avoid splitting words at chunk boundary? Actually no — just tokenize directly */
        int n_tokens = wubu_tokenizer_encode(&tok, chunk, tmp_ids, MAX_TOKENS_PER_CHUNK);
        if (n_tokens < 0) {
            fprintf(stderr, "FAILED: wubu_tokenizer_encode at pos %ld\n", pos);
            free(chunk); free(tmp_ids); free(corpus); fclose(fout);
            wubu_tokenizer_free(&tok);
            return 1;
        }

        /* Append tokens to growing array */
        if (total_tokens + n_tokens > all_tokens_cap) {
            all_tokens_cap = all_tokens_cap ? all_tokens_cap * 2 : 1048576;
            while (total_tokens + n_tokens > all_tokens_cap) all_tokens_cap *= 2;
            int *new_arr = (int *)realloc(all_tokens, all_tokens_cap * sizeof(int));
            if (!new_arr) {
                fprintf(stderr, "FAILED: realloc for %zu tokens\n", all_tokens_cap);
                free(chunk); free(tmp_ids); free(corpus); fclose(fout);
                wubu_tokenizer_free(&tok);
                return 1;
            }
            all_tokens = new_arr;
        }
        memcpy(all_tokens + total_tokens, tmp_ids, n_tokens * sizeof(int));
        total_tokens += n_tokens;

        pos += this_chunk;
        if (pos % 500000 == 0 || pos == fsize) {
            fprintf(stderr, "  Progress: %ld/%ld bytes, %zu tokens so far\n", pos, fsize, total_tokens);
        }
    }

    fprintf(stderr, "\nTotal tokens: %zu\n", total_tokens);

    /* Write binary uint32_t */
    fprintf(stderr, "Writing %s...\n", OUTPUT_PATH);
    size_t written = fwrite(all_tokens, sizeof(uint32_t), total_tokens, fout);
    if (written != total_tokens) {
        fprintf(stderr, "FAILED: wrote %zu / %zu uint32_t values\n", written, total_tokens);
    } else {
        fprintf(stderr, "  OK — wrote %zu uint32_t values (%zu bytes)\n", written, written * sizeof(uint32_t));
    }
    fclose(fout);

    /* Write metadata */
    fprintf(stderr, "Writing %s...\n", META_PATH);
    FILE *fmeta = fopen(META_PATH, "w");
    if (fmeta) {
        fprintf(fmeta, "vocab_size=%d\n", tok.vocab_size);
        fprintf(fmeta, "n_merges=%d\n", tok.n_merges);
        fprintf(fmeta, "bos_id=%d\n", tok.bos_id);
        fprintf(fmeta, "eos_id=%d\n", tok.eos_id);
        fprintf(fmeta, "pad_id=%d\n", tok.pad_id);
        fprintf(fmeta, "total_tokens=%zu\n", total_tokens);
        fprintf(fmeta, "corpus_chars=%ld\n", fsize);
        fprintf(fmeta, "corpus_file=%s\n", corpus_path);
        fclose(fmeta);
        fprintf(stderr, "  OK\n");
    }

    /* Cleanup */
    free(chunk);
    free(tmp_ids);
    free(all_tokens);
    free(corpus);
    wubu_tokenizer_free(&tok);

    fprintf(stderr, "Done.\n");
    return 0;
}
