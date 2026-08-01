/*
 * wubu_lookahead.h -- Lookahead / n-gram speculative helper (M06). Opaque-free.
 */
#ifndef WUBU_LOOKAHEAD_H
#define WUBU_LOOKAHEAD_H

/* Probe history for an n-gram ending at pos-1 that repeats earlier; return the
 * following token, or -1 if none found. */
int wubu_lookahead_probe(const int *history, int hist_len, int pos, int n);

#endif /* WUBU_LOOKAHEAD_H */
