/*
 * wubu_dgm.c -- DGM empirical gate + regression test runner (AX01). C11.
 *
 * Convergence (7-hop KB sweep: self-modifying safety, DGM, regression
 * testing, ReVeal, CoEvoSkills):
 *   - AX01: DGM empirical gate -- verified=1 only when gen_text returns 0
 *     AND oom_safe AND regression tests pass (anti-fake-log extended).
 *   - AX01b: regression test runner -- runs test_all before committing any
 *     self-modification; refuses to commit if tests regress.
 */
#include "wubu_dgm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WUBU_DGM_MAX_NODES 1024
#define WUBU_DGM_MAX_LINE 1024

/* ---- AX01: DGM empirical gate ---- */
int wubu_dgm_gate(const wubu_dgm_t *dgm, int gen_text_rc, int oom_safe,
                  int regression_passed) {
    if (!dgm) return 0;
    if (gen_text_rc != 0) return 0;
    if (!oom_safe) return 0;
    if (!regression_passed) return 0;
    return 1;
}

/* ---- AX01b: regression test runner ---- */
int wubu_dgm_regression_run(const char *test_cmd, char *out_buf, int buf_size) {
    if (!test_cmd || !out_buf || buf_size <= 0) return -1;
    FILE *fp = popen(test_cmd, "r");
    if (!fp) return -1;
    int total = 0, passed = 0, failed = 0;
    char line[WUBU_DGM_MAX_LINE];
    while (fgets(line, sizeof(line), fp)) {
        total++;
        if (strstr(line, "PASSED") || strstr(line, "ALL .*PASSED")) passed++;
        else if (strstr(line, "FAILED")) failed++;
    }
    int rc = pclose(fp);
    if (out_buf && buf_size > 0)
        snprintf(out_buf, buf_size, "regression: total=%d passed=%d failed=%d rc=%d",
                 total, passed, failed, rc);
    return (failed == 0 && rc == 0) ? 1 : 0;
}

/* ---- DGM archive helpers ---- */
int wubu_dgm_init(wubu_dgm_t *dgm) {
    if (!dgm) return -1;
    dgm->n_nodes = 0;
    dgm->cursor = 0;
    return 0;
}

int wubu_dgm_record(wubu_dgm_t *dgm, const char *variant_id, int verified,
                    double tok_s, int oom_safe) {
    if (!dgm || !variant_id || dgm->n_nodes >= WUBU_DGM_MAX_NODES) return -1;
    wubu_dgm_node_t *n = &dgm->nodes[dgm->n_nodes++];
    snprintf(n->variant_id, sizeof(n->variant_id), "%s", variant_id);
    n->verified = verified;
    n->tok_s = tok_s;
    n->oom_safe = oom_safe;
    n->timestamp = time(NULL);
    return 0;
}

const wubu_dgm_node_t *wubu_dgm_best(const wubu_dgm_t *dgm) {
    if (!dgm || dgm->n_nodes == 0) return NULL;
    int best = -1;
    double best_tok = -1.0;
    for (int i = 0; i < dgm->n_nodes; i++) {
        if (dgm->nodes[i].verified && dgm->nodes[i].tok_s > best_tok) {
            best_tok = dgm->nodes[i].tok_s;
            best = i;
        }
    }
    return (best >= 0) ? &dgm->nodes[best] : NULL;
}

int wubu_dgm_count_verified(const wubu_dgm_t *dgm) {
    if (!dgm) return 0;
    int c = 0;
    for (int i = 0; i < dgm->n_nodes; i++)
        if (dgm->nodes[i].verified) c++;
    return c;
}