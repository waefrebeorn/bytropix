/* wubu_epcap.c -- the episode-length cap. */
#include "wubu_epcap.h"

int wubu_epcap(const int *cost, int n, int budget, int *out)
{
    if (!cost || !out || n < 1 || budget < 1) return 0;
    int sum = 0, kept = 0;
    for (int i = 0; i < n; i++) {
        if (cost[i] < 0) return 0;
        if (sum + cost[i] > budget) break;
        sum += cost[i];
        kept++;
    }
    *out = kept;
    return kept == n;
}
