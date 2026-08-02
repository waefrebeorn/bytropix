/*
 * wubu_active.h -- Active Learning (uncertainty sampling / query-by-committee) (FF05).
 */
#ifndef WUBU_ACTIVE_H
#define WUBU_ACTIVE_H

#define WUBU_ACTIVE_MAX_PTS 128

typedef struct {
    int n;                       /* pool size */
    double mean[WUBU_ACTIVE_MAX_PTS];
    double var[WUBU_ACTIVE_MAX_PTS];   /* model variance per point */
    int   committee_disagree[WUBU_ACTIVE_MAX_PTS]; /* QBC disagreement */
    int   queried[WUBU_ACTIVE_MAX_PTS];
} wubu_active_t;

/* Uncertainty sampling: pick argmax variance among unqueried. */
int wubu_active_uncertainty(const wubu_active_t *al, int *out_idx);
/* Query-by-committee: pick argmax disagreement among unqueried. */
int wubu_active_qbc(const wubu_active_t *al, int *out_idx);
/* Mark a point as queried. */
int wubu_active_query(wubu_active_t *al, int idx);

#endif