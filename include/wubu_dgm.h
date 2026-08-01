/*
 * wubu_dgm.h -- DGM empirical gate + regression test runner (AX01).
 */
#ifndef WUBU_DGM_H
#define WUBU_DGM_H

#include <time.h>

#define WUBU_DGM_MAX_NODES 1024

typedef struct {
    char variant_id[128];
    int verified;
    double tok_s;
    int oom_safe;
    time_t timestamp;
} wubu_dgm_node_t;

typedef struct {
    wubu_dgm_node_t nodes[WUBU_DGM_MAX_NODES];
    int n_nodes;
    int cursor;
} wubu_dgm_t;

int wubu_dgm_init(wubu_dgm_t *dgm);
int wubu_dgm_record(wubu_dgm_t *dgm, const char *variant_id,
                          int verified, double tok_s, int oom_safe);
const wubu_dgm_node_t *wubu_dgm_best(const wubu_dgm_t *dgm);
int wubu_dgm_count_verified(const wubu_dgm_t *dgm);

/* AX01b: regression test runner */
int wubu_dgm_regression_run(const char *test_cmd, char *out_buf, int buf_size);

/* AX01c: DGM empirical gate (extends anti-fake-log) */
int wubu_dgm_gate(const wubu_dgm_t *dgm, int gen_text_rc,
                       int oom_safe, int regression_passed);

#endif