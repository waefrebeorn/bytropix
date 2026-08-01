/*
 * wubu_symbolic.h -- Symbolic rule engine + constraint checker (AW05, AW07).
 */
#ifndef WUBU_SYMBOLIC_H
#define WUBU_SYMBOLIC_H

#define WUBU_KB_MAX_FACTS 256

typedef struct {
    int pred;          /* predicate id */
    int args[3];       /* up to 3 integer args; -1 = wildcard in query */
} wubu_fact_t;

typedef struct {
    int n_facts;
    wubu_fact_t facts[WUBU_KB_MAX_FACTS];
} wubu_kb_t;

typedef struct {
    int head_pred;
    int head_args[3];    /* >=0 literal; <0 copies body arg (-1->0, -2->1, -3->2) */
    int body_pred;
    int body[3];         /* body args; -1 = wildcard */
} wubu_rule_t;

int  wubu_fact_add(wubu_kb_t *kb, int pred, int a0, int a1, int a2);
int  wubu_fact_query(const wubu_kb_t *kb, int pred, int a0, int a1, int a2);
int  wubu_rule_apply(wubu_kb_t *kb, const wubu_rule_t *r);
int  wubu_rules_run(wubu_kb_t *kb, const wubu_rule_t *rules, int n_rules);

/* AW05: constraint checker (default-deny). 998 = permitted, 999 = forbidden. */
int  wubu_constraint_permits(const wubu_kb_t *kb, int act_pred, int a0, int a1, int a2);

#endif
