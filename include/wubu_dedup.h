/* wubu_dedup.h -- the exact-duplicate-window scanner for the token
 * streams (the corpus-health tooling from the agentic-corpus wave): a
 * rolling hash finds the duplicate fixed-size token windows and reports
 * the duplication rate -- the raw material for the AC-B curation stage. */
#ifndef WUBU_DEDUP_H
#define WUBU_DEDUP_H

#include <stdint.h>

/* Scan a token stream for the duplicate windows of `win` tokens.
 * toks [n]: the token stream; win: the window size (>= 8).
 * dup [n] (may be NULL): 1 = the window STARTING at i is a duplicate of
 *   an earlier window (a bit per position).
 * Returns the number of duplicate windows found. */
long wubu_dedup_scan(const uint16_t *toks, long n, int win, uint8_t *dup);

/* The duplicate rate (duplicate windows / total windows, 0 when n < win). */
float wubu_dedup_rate(const uint8_t *dup, long n, int win);

#endif
