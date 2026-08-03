/*
 * barun_tokenc.c -- the C11 corpus tokenizer (the AGI corpus pipeline).
 *
 * Reads text files (or stdin), tokenizes with the byte-level BPE
 * tokenizer (wubu_tok_hf -- OUR implementation), and writes compact
 * uint16 token streams: <bos> doc <eos> per document, little-endian.
 * The wubuwizard trainer consumes these .tok streams directly.
 *
 * Usage:
 *   barun_tokenc <tokenizer.json> <input.txt|-> <output.tok>
 */
#define _POSIX_C_SOURCE 200809L   /* getline is POSIX, not C11 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu_tokenizer_hf.h"

static void write_u16(FILE *f, uint16_t v)
{
    fwrite(&v, sizeof(v), 1, f);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s <tokenizer.json> <input.txt|-> <output.tok>\n",
                argv[0]);
        return 2;
    }
    const char *tok_path = argv[1];
    const char *in_path = argv[2];
    const char *out_path = argv[3];

    wubu_tok_hf_t *tok = wubu_tok_hf_load(tok_path);
    if (!tok) {
        fprintf(stderr, "tokenc: cannot load %s\n", tok_path);
        return 1;
    }
    FILE *in = strcmp(in_path, "-") == 0 ? stdin : fopen(in_path, "r");
    if (!in) { fprintf(stderr, "tokenc: cannot open %s\n", in_path); return 1; }
    FILE *out = fopen(out_path, "wb");
    if (!out) { fprintf(stderr, "tokenc: cannot open %s\n", out_path); return 1; }

    int bos = wubu_tok_hf_bos_id(tok);
    int eos = wubu_tok_hf_eos_id(tok);
    if (bos < 0) bos = 2;
    if (eos < 0) eos = 3;

    /* stream: read lines, group into documents on blank lines */
    char *line = NULL;
    size_t cap = 0;
    int ids[16384];
    long n_docs = 0, n_tokens = 0;
    int in_doc = 0;
    while (getline(&line, &cap, in) >= 0) {
        /* strip the newline */
        size_t len = strlen(line);
        while (len && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;
        int is_blank = 1;
        for (size_t i = 0; i < len; i++)
            if (line[i] != ' ' && line[i] != '\t') { is_blank = 0; break; }
        if (is_blank) {
            if (in_doc) { write_u16(out, (uint16_t)eos); in_doc = 0; n_tokens++; }
            continue;
        }
        if (!in_doc) { write_u16(out, (uint16_t)bos); in_doc = 1; n_docs++; n_tokens++; }
        /* chunk very long lines: the BPE merge loop is O(n^2) per call,
         * so a 10K-char paragraph would stall; 256-char chunks keep the
         * merge cost bounded while preserving the token context. */
        size_t off = 0;
        while (off < len) {
            size_t take = len - off;
            if (take > 256) take = 256;
            /* cut at a space boundary when possible */
            if (off + take < len) {
                size_t cut = take;
                while (cut > 32 && line[off + cut - 1] != ' ') cut--;
                if (cut > 32) take = cut;
            }
            char chunk[512];
            memcpy(chunk, line + off, take);
            chunk[take] = 0;
            int n = wubu_tok_hf_encode(tok, chunk, ids, 16384);
            if (n > 0) {
                for (int i = 0; i < n; i++) write_u16(out, (uint16_t)ids[i]);
                n_tokens += n;
            }
            off += take;
        }
        /* a small end-of-document heuristics: a line ending with '.', '?',
         * '!' or ':' closes the paragraph-ish doc */
        if (len && (line[len-1] == '.' || line[len-1] == '!' ||
                    line[len-1] == '?' || line[len-1] == ':')) {
            if (len < 200) { write_u16(out, (uint16_t)eos); in_doc = 0; n_tokens++; }
        }
    }
    if (in_doc) { write_u16(out, (uint16_t)eos); n_tokens++; }
    free(line);
    fclose(in);
    fclose(out);
    fprintf(stderr, "tokenc: %ld docs, %ld tokens -> %s\n",
            n_docs, n_tokens, out_path);
    wubu_tok_hf_free(tok);
    return 0;
}
