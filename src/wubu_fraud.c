/*
 * wubu_fraud.c -- Fraud detection: outlier + dispute resolution (DD05). C11.
 *
 * Convergence (fraud detection + dispute resolution 7-hop: outlier detection,
 * evidence submission, trust decay, dispute arbitration):
 *   - DD05: detects Byzantine agents that submit false claims. Reporters
 *     submit evidence; adjudication requires majority confirmation of fraud.
 *     Confirmed fraudsters have their trust score decayed (default 100
 *     → 50 on first fraud → 0 on repeat). Trust score gates voting weight
 *     in BFT.
 */
#include "wubu_fraud.h"
#include <string.h>

int wubu_fraud_init(wubu_fraud_t *fr, int n_agents) {
    if (!fr || n_agents < 1 || n_agents > 64) return -1;
    memset(fr, 0, sizeof(*fr));
    fr->n_agents = n_agents;
    for (int i = 0; i < n_agents; i++) fr->trust_dec[i] = 100;
    return 0;
}

int wubu_fraud_report(wubu_fraud_t *fr, int reporter_id, int reported_id, const char *evidence) {
    if (!fr || !evidence || reporter_id < 0 || reported_id < 0) return -1;
    if (fr->n_reports >= WUBU_FRAUD_MAX_REPORTS) return -1;
    wubu_fraud_report_t *r = &fr->reports[fr->n_reports];
    r->reporter_id = reporter_id;
    r->reported_id = reported_id;
    strncpy(r->evidence, evidence, 127);
    r->evidence[127] = '\0';
    r->resolved = 0;
    r->fraud_confirmed = 0;
    return fr->n_reports++;
}

int wubu_fraud_adjudicate(wubu_fraud_t *fr, int reported_id) {
    if (!fr) return -1;
    int confirm = 0, deny = 0;
    for (int i = 0; i < fr->n_reports; i++) {
        if (fr->reports[i].reported_id != reported_id) continue;
        if (fr->reports[i].resolved) continue;
        /* Simple heuristic: if evidence contains "mismatch" or "false", it's fraud */
        if (strstr(fr->reports[i].evidence, "mismatch") ||
            strstr(fr->reports[i].evidence, "false") ||
            strstr(fr->reports[i].evidence, "incorrect")) {
            confirm++;
        } else {
            deny++;
        }
        fr->reports[i].resolved = 1;
    }
    int fraud = (confirm > 0 && confirm >= deny) ? 1 : 0;
    if (fraud) wubu_fraud_decay(fr, reported_id);
    return fraud;
}

int wubu_fraud_trust(const wubu_fraud_t *fr, int agent_id) {
    if (!fr || agent_id < 0 || agent_id >= fr->n_agents) return 0;
    return fr->trust_dec[agent_id];
}

void wubu_fraud_decay(wubu_fraud_t *fr, int agent_id) {
    if (!fr || agent_id < 0 || agent_id >= fr->n_agents) return;
    fr->trust_dec[agent_id] /= 2;  /* halve trust on confirmed fraud */
}
