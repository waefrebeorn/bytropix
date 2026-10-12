/* wubu_data_mix.c — data mixture weighting (BRAIN #243).
 * C11, self-contained. p_i = w_i / sum(w); sampling via cumulative scan.
 */
#include "wubu_data_mix.h"

int wubu_data_mix(const double *weights, int k, double *out_props) {
    if (!weights || !out_props || k <= 0) return -1;
    double sum = 0.0;
    for (int i = 0; i < k; i++) {
        if (weights[i] < 0.0) return -1;
        sum += weights[i];
    }
    if (sum <= 0.0) return -1;
    for (int i = 0; i < k; i++) out_props[i] = weights[i] / sum;
    return 0;
}

int wubu_data_mix_sample(const double *props, int k, double u) {
    if (!props || k <= 0 || u < 0.0 || u >= 1.0) return -1;
    double acc = 0.0;
    for (int i = 0; i < k; i++) {
        acc += props[i];
        if (u < acc) return i;
    }
    return k - 1;   /* u in the last bucket (float rounding) */
}
