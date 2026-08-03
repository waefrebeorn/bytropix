/* wubu_mix.c -- the weighted multi-stream corpus mixer. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_mix.h"

long wubu_mix_build(const char **paths, const float *weights, int n,
                    uint16_t *out, long out_cap, int chunk)
{
    if (!paths || !weights || !out || n < 1 || out_cap < 1 || chunk < 1)
        return -1;
    /* open all the streams + read them entirely (the .tok streams are
     * bounded by the memory; the cap is the caller's) */
    FILE **f = (FILE **)calloc((size_t)n, sizeof(FILE *));
    long *len = (long *)calloc((size_t)n, sizeof(long));
    uint16_t **buf = (uint16_t **)calloc((size_t)n, sizeof(uint16_t *));
    if (!f || !len || !buf) return -1;
    long total = 0;
    for (int i = 0; i < n; i++) {
        f[i] = fopen(paths[i], "rb");
        if (!f[i]) { fprintf(stderr, "wubu_mix: cannot open %s\n", paths[i]); return -1; }
        fseek(f[i], 0, SEEK_END);
        long sz = ftell(f[i]) / 2;
        fseek(f[i], 0, SEEK_SET);
        buf[i] = (uint16_t *)malloc((size_t)sz * 2);
        if (!buf[i]) return -1;
        len[i] = 0;
        while (len[i] < sz && fread(&buf[i][len[i]], 2, 1, f[i]) == 1) len[i]++;
        fclose(f[i]);
        total += len[i];
    }
    /* the deterministic smooth weighted round-robin (nginx-style -- no
     * RNG at all: current[i] += weight[i]; pick the argmax; subtract the
     * total weight. The 3:1 mix interleaves as a,a,b,a,a,b,... -- fair
     * and reproducible. The RNG seeds kept picking only stream a for the
     * first ~16 draws (an unlucky 0.75^16), so the RNG is gone. */
    double *cur = (double *)calloc((size_t)n, sizeof(double));
    if (!cur) return -1;
    double wsum2 = 0;
    for (int i = 0; i < n; i++) wsum2 += weights[i] > 0 ? weights[i] : 0;
    long out_n = 0;
    long *pos = (long *)calloc((size_t)n, sizeof(long));
    if (!pos) return -1;
    long guard = 0, max_guard = (total / chunk + n) * 2 + 16;
    while (out_n < out_cap && guard++ < max_guard) {
        int pick = 0;
        for (int i = 0; i < n; i++) {
            cur[i] += weights[i] > 0 ? weights[i] : 0;
            if (cur[i] > cur[pick]) pick = i;
        }
        cur[pick] -= wsum2;
        if (pos[pick] >= len[pick]) continue;   /* exhausted -- try again */
        long take = chunk;
        if (take > len[pick] - pos[pick]) take = len[pick] - pos[pick];
        if (take > out_cap - out_n) take = out_cap - out_n;
        memcpy(out + out_n, buf[pick] + pos[pick], (size_t)take * 2);
        out_n += take;
        pos[pick] += take;
    }
    free(pos); free(cur);
    for (int i = 0; i < n; i++) free(buf[i]);
    free(f); free(len); free(buf);
    return out_n;
}
