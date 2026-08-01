/*
 * wubu_metagame2.h -- Deeper meta-game primitives (AH07/AH09/AH10/AH11).
 */
#ifndef WUBU_METAGAME2_H
#define WUBU_METAGAME2_H

#include <math.h>

#define WUBU_SKILL_MAX 128

typedef struct {
    char name[WUBU_SKILL_MAX][32];
    char body[WUBU_SKILL_MAX][256];
    double score[WUBU_SKILL_MAX];
    int n;
} wubu_skilllib_t;

typedef struct {
    long *buf;
    int  cap;
    int  n;
    long seen;
} wubu_replay_t;

typedef struct {
    double calib;  /* EMA calibration error (|conf-actual|) */
    int n;
} wubu_metacog_t;

int  wubu_sandbox_allow(unsigned int caps, int net_ok, int fs_ok, int tests_ok);
int  wubu_skill_add(wubu_skilllib_t *s, const char *name, const char *body, double score);
int  wubu_skill_topk(const wubu_skilllib_t *s, int k, int *out);
int  wubu_replay_add(wubu_replay_t *r, long exp_id, int *replace_idx);
double wubu_metacog_update(wubu_metacog_t *m, double confidence, int correct);
int  wubu_metacog_calibrated(const wubu_metacog_t *m, double thr);

#endif
