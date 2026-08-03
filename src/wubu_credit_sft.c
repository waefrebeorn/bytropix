/* wubu_credit_sft.c -- the credit-assignment SFT mask. The doctrine
 * (Orchard): a trajectory that never resolved still contains productive
 * segments -- the leading run of successes. The mask credits the longest
 * prefix of successful steps (and a final isolated success before the
 * failure tail), so the model learns the good prefix without learning the
 * failure. */
#include "wubu_credit_sft.h"

int wubu_credit_mask(const int *succ, int n, int *mask)
{
    if (!succ || n < 1 || !mask) return 0;
    int credited = 0;
    /* the leading run of successes */
    int i = 0;
    while (i < n && succ[i]) { mask[i] = 1; credited++; i++; }
    /* the failure tail: zero */
    for (; i < n; i++) mask[i] = 0;
    /* a single isolated success right before the first failure is part of
     * the productive prefix already (it IS the leading run's end); the
     * trailing successes after a failure are NOT credited (the model
     * should not learn recovery it did not do) */
    return credited;
}

float wubu_credit_frac(const int *mask, int n)
{
    if (!mask || n < 1) return 0;
    int c = 0;
    for (int i = 0; i < n; i++) if (mask[i]) c++;
    return (float)c / (float)n;
}
