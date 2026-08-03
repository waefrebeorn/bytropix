/* wubu_dedup_cli.c -- the corpus-health scanner: measures the exact
 * duplicate-window rate of a .tok stream (the AC-B curation stage).
 * Usage: wubu_dedup_cli <file.tok> [window] [max-tokens]
 * The window defaults to 4096; the scan reads at most max-tokens
 * (default 200M -- the whole 7.5GB stream is a multi-minute scan). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_dedup.h"

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <file.tok> [window] [max-tokens]\n", argv[0]); return 1; }
    const char *path = argv[1];
    int win = argc > 2 ? atoi(argv[2]) : 4096;
    long maxn = argc > 3 ? atol(argv[3]) : 200000000L;

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    uint16_t *buf = (uint16_t *)malloc((size_t)maxn * 2);
    if (!buf) { fclose(f); return 1; }
    long n = 0;
    while (n < maxn && fread(&buf[n], 2, 1, f) == 1) n++;
    fclose(f);
    if (n < win) { fprintf(stderr, "stream too short (%ld)\n", n); free(buf); return 1; }

    uint8_t *dup = (uint8_t *)calloc((size_t)n, 1);
    if (!dup) { free(buf); return 1; }
    long ndup = wubu_dedup_scan(buf, n, win, dup);
    float rate = wubu_dedup_rate(dup, n, win);
    printf("dedup: %s | %ld tokens, win %d, %ld dup windows, rate %.4f\n",
           path, n, win, ndup, rate);
    free(dup);
    free(buf);
    return 0;
}
