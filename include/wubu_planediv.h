/*
 * wubu_planediv.h -- Control/data-plane separation + poisoning divergence (AG02/AG03).
 */
#ifndef WUBU_PLANEDIV_H
#define WUBU_PLANEDIV_H

#define WUBU_PLANE_CONTROL 0
#define WUBU_PLANE_DATA    1

typedef struct {
    int allow_data_as_instruction; /* default 0 = deny data-plane as instruction */
} wubu_plane_t;

int  wubu_plane_enforce(const wubu_plane_t *p, int item_plane, const char *content);
unsigned long long wubu_mem_fingerprint(const char *blob, int n);
int  wubu_mem_diverged(unsigned long long trusted_fp, unsigned long long cur_fp);
int  wubu_replay_flagged(unsigned long long fp, const unsigned long long *seen, int n_seen);

#endif
