#ifndef WUBU_EMBEDDING_NORM_H
#define WUBU_EMBEDDING_NORM_H
/* wubu_embedding_norm.h — embedding scaling (BRAIN #238).
 *
 * Standard transformer embedding scale: multiply token embeddings by sqrt(d)
 * so their variance matches the ~unit-variance attention/positional signals.
 * The forward (wubu_embed_scale_factor + wubu_embed_scale) is covered by
 * test_embedding_norm with measured assertions.
 */
#include <stddef.h>

/* The embedding scale factor sqrt(d). Returns 0.0 if d<=0. */
double wubu_embed_scale_factor(int d);

/* Scale `n` embeddings (row-major, each of width d) by sqrt(d) in place.
 * Returns 0 ok, -1 bad args. */
int wubu_embed_scale(float *emb, int n, int d);

#endif
