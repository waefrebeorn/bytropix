/*
 * wubu_lookahead.c -- Lookahead / n-gram speculative helper (M06). C11.
 *
 * Convergence (lookahead decoding / n-gram speculative 7-hop): instead of a
 * draft model, scan recent token history for a repeated n-gram and propose the
 * token that followed it before. Cheap, training-free speculation. Given a
 * history buffer and a current position, find the longest suffix match and return
 * the next token (or -1 if none). Triple-DA: hist==NULL/hist_len<=0/n<=0 -> -1;
 * no OOB; deterministic (first match wins).
 */
#include "wubu_lookahead.h"
#include <stdlib.h>

/* Look for an n-gram of length `n` ending at history position `pos-1` that also
 * occurs earlier; return the token following the earlier occurrence, else -1.
 * history is int array of length hist_len; pos is the current write head. */
int wubu_lookahead_probe(const int *history, int hist_len, int pos, int n) {
    if (!history || hist_len <= 0 || n <= 0) return -1;
    if (pos < n || pos > hist_len) return -1;
    if (n > pos) n = pos;                 /* clamp n to available */
    /* target suffix = history[pos-n .. pos-1] */
    for (int start = 0; start + n < pos; start++) {
        int ok = 1;
        for (int i = 0; i < n; i++)
            if (history[start + i] != history[pos - n + i]) { ok = 0; break; }
        if (ok) {
            /* found earlier occurrence ending at start+n-1; token after it */
            int after = start + n;
            if (after < pos) return history[after];
            return -1;
        }
    }
    return -1;
}
