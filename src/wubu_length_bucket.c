/* wubu_length_bucket.c — length bucketing (BRAIN #246).
 * C11, self-contained. First ascending bound >= L wins; overflow -> last bucket.
 */
#include "wubu_length_bucket.h"

int wubu_length_bucket(int L, const int *bounds, int k) {
    if (!bounds || k <= 0) return -1;
    if (L <= 0) return 0;
    for (int i = 0; i < k; i++)
        if (L <= bounds[i]) return i;
    return k - 1;   /* longer than all bounds -> last bucket */
}

int wubu_length_bucket_pad(int L, const int *bounds, int k) {
    int b = wubu_length_bucket(L, bounds, k);
    if (b < 0) return -1;
    return bounds[b];
}
