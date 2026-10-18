/*
 * wubu_decision.h -- typed decision node (the SystemOne/Clef/Jev contract).
 *
 * A decision model consumes a state plus a schema of typed questions and
 * returns a probability for every allowed option, with no free-form text. That
 * is the right shape for an OS control plane: the kernel can act on a number.
 *
 * Two things the hosted models do NOT guarantee, and this module does:
 *   - per-question probabilities sum to 1 (softmax is exact, not approximate);
 *   - logically related questions are coherent (arXiv 2609.33209 measured Jev's
 *     "X" and "not X" probabilities missing 1 by 0.064 on average, and three
 *     single-label probabilities summing to 1.14).
 *
 * The gate is deliberately separate from the model: a model can be swapped,
 * the arithmetic contract cannot.
 */
#ifndef WUBU_DECISION_H
#define WUBU_DECISION_H

/* Softmax over n logits. Returns 0, or -1 on bad arguments. */
int wubu_decision_softmax(const float *logits, int n, float *probs);

/* Shannon entropy of a distribution (nats). -1.0 on bad arguments. */
float wubu_decision_entropy(const float *probs, int n);

/* Typed verdict. Returns 1 when max(probs) >= threshold (act), else 0 (defer).
 * *idx receives the argmax, *conf the max probability, on both paths. */
int wubu_decision_choose(const float *probs, int n, float threshold,
                         int *idx, float *conf);

/* Complement consistency: |p(x) + p(not x) - 1| <= tol. Returns 1 if coherent. */
int wubu_decision_complement_ok(float p, float p_not, float tol);

/* Mutual exclusivity: |sum(probs) - 1| <= tol. Returns 1 if coherent. */
int wubu_decision_exclusive_ok(const float *probs, int n, float tol);

/* Expected calibration error over confidence/correctness pairs, `bins` bins.
 * Returns the ECE in [0,1], or -1 on bad arguments. */
float wubu_decision_ece(const float *conf, const unsigned char *correct,
                        int n, int bins);

/* ---- Delegation economics (JevAgentHarness, arXiv 2610.21412) ----------
 *
 * A decision model is only worth consulting when its confidence clears a
 * threshold that pays for the risk. With confidence == accuracy the break-even
 * point is  tau = q_L - dC / V  (LLM accuracy minus the saved call's price
 * divided by the cost of a wrong choice). The fitter below does not assume
 * that equality: it scans the agent's own shadow-mode samples and picks the
 * cheapest threshold whose projected success stays within `budget` of the LLM.
 */
typedef struct {
    float conf;            /* decision model's confidence on this state */
    unsigned char correct; /* 1 if its answer matched the answer key */
} wubu_decision_sample;

/* Analytic break-even threshold. Returns tau in [0,1], or -1 on bad args. */
double wubu_decision_analytic_threshold(double llm_acc, double saved_cost,
                                        double wrong_cost);

/* Fit a threshold from shadow-mode samples.
 *   llm_acc     how often the LLM gets this decision right (q_L)
 *   cost_probe  price of one decision-model call
 *   cost_llm    price of the LLM call a delegation would save
 *   cost_wrong  cost of one wrong accepted answer (V)
 *   budget      allowed drop in success (e.g. 0.03 for 3 points)
 * Returns the threshold, 2.0 when delegating nothing is cheapest, or -1 when
 * there is no sample to fit on. */
double wubu_decision_fit_threshold(const wubu_decision_sample *s, int n,
                                   double llm_acc, double cost_probe,
                                   double cost_llm, double cost_wrong,
                                   double budget);

/* Per-decision-point accounting: what was delegated, what it cost, what it
 * saved. The paper's harness prices every call; so does this. */
typedef struct {
    unsigned long delegated;
    unsigned long deferred;
    unsigned long wrong_delegated;
    double cost_dm;
    double cost_llm_spent;   /* LLM calls still made on deferrals */
    double cost_llm_avoided; /* LLM calls the delegations replaced */
    double cost_penalty;     /* wrong_delegated * V, if the caller sets V */
    double time_dm_s;
    double time_llm_s;       /* time spent in LLM calls actually made */
    double time_avoided_s;   /* LLM time the delegations replaced */
} wubu_decision_ledger;

void wubu_decision_ledger_init(wubu_decision_ledger *l);

/* Record one decision. `delegated` = 0 means the LLM decided. */
void wubu_decision_ledger_record(wubu_decision_ledger *l, int delegated,
                                 int correct, double dm_cost, double dm_time,
                                 double llm_cost, double llm_time,
                                 double wrong_cost);

/* Total cost = dm + llm spent + penalties (costs avoided are reported, not
 * subtracted). Returns -1.0 on a NULL ledger. */
double wubu_decision_ledger_cost(const wubu_decision_ledger *l);

/* Percent of a baseline (LLM-always) cost saved by the delegations.
 * Returns 0 when the baseline is not positive. */
double wubu_decision_ledger_saved_pct(const wubu_decision_ledger *l,
                                      double baseline_cost);


#endif /* WUBU_DECISION_H */