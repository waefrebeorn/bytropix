/*
 * wubu_metagame.h -- Open-ended self-modifying agent archive (AH05/AH06/AH08/AH13).
 */
#ifndef WUBU_METAGAME_H
#define WUBU_METAGAME_H

#define WUBU_ARCHIVE_MAX 256

typedef struct {
    char id[WUBU_ARCHIVE_MAX][32];
    char parent[WUBU_ARCHIVE_MAX][32];
    double fitness[WUBU_ARCHIVE_MAX];
    int   verified[WUBU_ARCHIVE_MAX];
    int   n;
} wubu_archive_t;

int   wubu_archive_add(wubu_archive_t *a, const char *id, const char *parent,
                       double fitness, int verified);
int   wubu_accept_child(const wubu_archive_t *a, const char *child_id,
                       double child_fit, int verified);
int   wubu_improvement_delta(double child, double parent, double min_gain);
double wubu_archive_best(const wubu_archive_t *a);

#endif
