/*
 * wubu_fraud.h -- Fraud detection (outlier + dispute resolution) (DD05).
 */
#ifndef WUBU_FRAUD_H
#define WUBU_FRAUD_H

#define WUBU_FRAUD_MAX_REPORTS 64

typedef struct {
    int   reporter_id;
    int   reported_id;
    char  evidence[128];
    int   resolved;  /* 0 = pending, 1 = fraud confirmed, -1 = false report */
    int   fraud_confirmed;
} wubu_fraud_report_t;

typedef struct {
    wubu_fraud_report_t reports[WUBU_FRAUD_MAX_REPORTS];
    int n_reports;
    int trust_dec[64];  /* trust score per agent (decays on fraud) 0-100 */
    int n_agents;
} wubu_fraud_t;

int  wubu_fraud_init(wubu_fraud_t *fr, int n_agents);
/* Report a suspected fraud. Returns report index or -1. */
int  wubu_fraud_report(wubu_fraud_t *fr, int reporter_id, int reported_id, const char *evidence);
/* Adjudicate: check if evidence from majority of reporters confirms fraud. */
int  wubu_fraud_adjudicate(wubu_fraud_t *fr, int reported_id);
/* Get trust score for an agent (0-100). */
int  wubu_fraud_trust(const wubu_fraud_t *fr, int agent_id);
/* Decay trust of a confirmed fraudster. */
void wubu_fraud_decay(wubu_fraud_t *fr, int agent_id);

#endif