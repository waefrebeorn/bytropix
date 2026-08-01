/*
 * wubu_resource.h -- At-home resource envelope (AH14/AH15).
 */
#ifndef WUBU_RESOURCE_H
#define WUBU_RESOURCE_H

enum { WUBU_TIER_FIT = 0, WUBU_TIER_FITS_Q3, WUBU_TIER_NOFIT };

int   wubu_pick_tier(double vram_gb, int billions);
double wubu_est_toks(double bandwidth_gbs, int billions, int quant_bits);
int   wubu_degrade_tier(double vram_gb, int desired_b);

#endif
