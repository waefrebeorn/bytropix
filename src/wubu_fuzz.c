/*
 * wubu_fuzz.c -- robustness / fuzzing frontier (Theme IX). C11.
 */
#include "wubu_fuzz.h"
#include <string.h>
#include <ctype.h>

int wubu_fuzz_mutate(const char *in, char *out, int cap, uint32_t seed)
{
    if (!in || !out || cap <= 0) return -1;
    int n = (int)strlen(in);
    if (n >= cap - 1) n = cap - 2;
    uint32_t rng = seed ? seed : 1;
    for (int i = 0; i < n; i++) {
        rng = rng * 1664525u + 1013904223u;
        char c = in[i];
        switch (rng % 4) {
        case 0: if (c == ' ') c = '_'; break;
        case 1: if (c >= 'a' && c <= 'z') c = (char)(c - 32); break;
        case 2: if (c == '!') c = '.'; break;
        default: break;
        }
        out[i] = c;
    }
    out[n] = 0;
    return n;
}

float wubu_fuzz_evasion(long evaded, long total)
{
    if (total <= 0) return 0;
    return (float)evaded / (float)total;
}

int wubu_fuzz_sensitivity(const char *prompt, const char *forbidden,
                          int *distance)
{
    if (!prompt || !forbidden || !distance) return -1;
    /* the first occurrence distance of the forbidden substring */
    const char *hit = strstr(prompt, forbidden);
    if (!hit) { *distance = -1; return 0; }
    *distance = (int)(hit - prompt);
    return 1;
}

int wubu_fuzz_crash_valid(int segv, int oom, int timeout, int reachable)
{
    /* a crash is a real bug only if reachable + not an environmental */
    if (!reachable) return 0;
    if (segv) return 1;
    if (oom) return 0;    /* memory-cap artifacts */
    if (timeout) return 1;
    return 0;
}

float wubu_fuzz_divergence(const char *a, const char *b)
{
    if (!a || !b) return 1.0f;
    int la = (int)strlen(a), lb = (int)strlen(b);
    if (la == 0 && lb == 0) return 0;
    /* the normalized edit-distance-ish: the shared-prefix fraction */
    int same = 0, n = la < lb ? la : lb;
    while (same < n && a[same] == b[same]) same++;
    float denom = (la > lb ? la : lb);
    return 1.0f - (float)same / (denom > 0 ? denom : 1.0f);
}

int wubu_fuzz_cov_mutate(const char *in, char *out, int cap,
                         const uint8_t *covered, int n)
{
    if (!in || !out || cap <= 0) return -1;
    int len = (int)strlen(in);
    if (len >= cap - 1) len = cap - 2;
    for (int i = 0; i < len; i++) {
        char c = in[i];
        if (covered && i < n && !covered[i]) {
            /* uncovered regions get mutated aggressively */
            c = (c == ' ') ? '\t' : ' ';
        }
        out[i] = c;
    }
    out[len] = 0;
    return len;
}

int wubu_fuzz_gate(float new_evasion, float old_evasion, float th)
{
    return (new_evasion > old_evasion + th) ? 1 : 0;
}

int wubu_fuzz_taxonomy(const char *prompt, int *bucket)
{
    if (!prompt || !bucket) return -1;
    /* coarse taxonomy: 0 direct, 1 roleplay, 2 encoded, 3 reframed */
    if (strstr(prompt, "ignore") || strstr(prompt, "bypass"))
        { *bucket = 1; return 1; }
    if (strstr(prompt, "base64") || strstr(prompt, "hex"))
        { *bucket = 2; return 1; }
    if (strstr(prompt, "pretend") || strstr(prompt, "imagine"))
        { *bucket = 3; return 1; }
    *bucket = 0;
    return 0;
}

int wubu_fuzz_validate(const char *in, int max_len, int has_newline,
                       int has_control)
{
    if (!in) return -1;
    int len = (int)strlen(in);
    if (max_len > 0 && len > max_len) return 0;
    if (has_newline && strchr(in, '\n')) return 0;
    if (has_control) {
        for (int i = 0; i < len; i++)
            if (iscntrl((unsigned char)in[i]) && in[i] != '\n') return 0;
    }
    return 1;
}

float wubu_fuzz_seed(float diversity, float past_yield)
{
    return diversity * (0.5f + 0.5f * past_yield);
}
