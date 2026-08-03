/* wubu_recency.c -- the recency-weighted sampling. */
#include <math.h>
#include "wubu_recency.h"

float wubu_recency_weight(long i, long n, float base, float power)
{
    if (n <= 0) return base;
    if (base < 0) base = 0;
    if (base > 1) base = 1;
    if (power < 0) power = 0;
    double f = (double)i / (double)n;
    if (f > 1) f = 1;
    return (float)(base + (1.0 - base) * pow(f, power));
}
