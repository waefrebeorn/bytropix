/* wubu_embedding_norm.c — embedding scaling (BRAIN #238).
 * C11, self-contained. Scale = sqrt(d); each embedding row multiplied by it.
 */
#include "wubu_embedding_norm.h"
#include <math.h>

double wubu_embed_scale_factor(int d) {
    if (d <= 0) return 0.0;
    return sqrt((double)d);
}

int wubu_embed_scale(float *emb, int n, int d) {
    if (!emb || n <= 0 || d <= 0) return -1;
    double s = wubu_embed_scale_factor(d);
    for (int i = 0; i < n * d; i++) emb[i] = (float)((double)emb[i] * s);
    return 0;
}
