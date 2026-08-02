/*
 * wubu_metacog.h -- AGI metacognition (Theme JD, first batch). C11.
 *
 * Convergence (metacognition survey 2607.11881; MetaCogAgent 2605.17292
 * ECE 0.087; metacognitive harness 2605.14186; Meta-R1):
 *   - Capability profiles per agent (calibrated competence)
 *   - Self-assessment before task execution
 *   - Expected-calibration-error (ECE) tracker
 *   - JOL (judgment-of-learning) extraction + inference-time harness
 *   - Monitor-actor separation (independent verifier)
 *   - Learning-progress prediction + calibration without ground truth
 *   - Verbalized-uncertainty faithfulness
 *   - MetaCog-Eval-style benchmark runner + competence-difficulty gap
 *   - Capability-weighted delegation + profile update loop
 *   - Second-order anomaly + calibration drift monitoring
 *   - Confidence-calibrated sampling + monitoring traces
 */
#ifndef WUBU_METACOG_H
#define WUBU_METACOG_H

#include <stdint.h>

#define WUBU_MC_MAX_AGENTS 16
#define WUBU_MC_MAX_BINS 16

/* JD01/JD02: capability profile + self-assessment. */
typedef struct {
    float competence[WUBU_MC_MAX_AGENTS];  /* calibrated competence */
    float confidence[WUBU_MC_MAX_AGENTS];  /* latest self-assessment */
    int n_agents;
    int trained;   /* profiles calibrated yet */
} wubu_metacog_t;

int   wubu_mc_init(wubu_metacog_t *m, int n_agents);
/* JD01: set/update a calibrated competence for an agent. */
int   wubu_mc_set_competence(wubu_metacog_t *m, int agent, float comp);
/* JD02: self-assessment -- the agent's confidence estimate, clamped to
 * the competence envelope (an overconfident agent gets pulled back). */
float wubu_mc_assess(const wubu_metacog_t *m, int agent, float claimed);

/* JD03: expected-calibration-error tracker. */
typedef struct {
    double conf_sum, acc_sum, conf2_sum;
    int n;
} wubu_mc_ece_t;

int   wubu_mc_ece_init(wubu_mc_ece_t *e);
int   wubu_mc_ece_feed(wubu_mc_ece_t *e, float confidence, float correct);
float wubu_mc_ece_score(const wubu_mc_ece_t *e);

/* JD04/JD05: JOL extraction + metacognitive harness. The JOL of a
 * token = the model's log-prob gap between the top-1 and top-2
 * candidates (a larger gap = higher judgment of learning). The harness
 * maps the JOL to an inference-time control: stop / retry / continue. */
float wubu_mc_jol(float logprob_top1, float logprob_top2);
int   wubu_mc_harness(float jol, float low, float high, int retries_left);

/* JD08: monitor-actor separation check. Returns 1 when the monitor's
 * prediction disagrees with the actor's (they've drifted apart). */
int wubu_mc_separated(float monitor_pred, float actor_pred, float drift);

/* JD09: learning-progress prediction -- the predicted competence after
 * k more trials, from the current competence and the per-trial gain. */
float wubu_mc_progress(float competence, float per_trial_gain, int k);

/* JD10: calibration without ground truth -- self-supervised
 * recalibration: when the model's confidence distribution flattens
 * (uncertainty creep), scale the confidence down. */
float wubu_mc_recalibrate(float confidence, float entropy, float entropy_base);

/* JD11: verbalized-uncertainty faithfulness -- the |spoken - actual|
 * gap; lower is more faithful. */
float wubu_mc_faithfulness(float verbalized, float actual);

/* JD12: MetaCog-Eval-style runner -- scores a batch of tasks against
 * the competence threshold. Returns the passed count. */
int wubu_mc_eval_run(const float *task_difficulties, int n,
                     float competence, float pass_thresh, int *passed);

/* JD13: capability-weighted delegation -- pick the agent whose
 * competence best covers the task difficulty. */
int wubu_mc_delegate(const wubu_metacog_t *m, float difficulty);

/* JD14/JD21: profile update loop -- competence drifts toward the
 * observed outcome (EMA). */
int wubu_mc_update_competence(wubu_metacog_t *m, int agent, float outcome,
                              float alpha);

/* JD16: monitoring trace -- append a monitoring event. */
int wubu_mc_trace(uint32_t *trace, int *n, int max, uint32_t event);

/* JD17: second-order anomaly -- the monitor flags its own drift. */
int wubu_mc_second_order(float monitor_score, float baseline, float thresh);

/* JD18: calibration telemetry -- the rolling calibration window mean. */
float wubu_mc_telemetry(const float *window, int n);

/* JD19: competence-difficulty gap -- positive = the agent can handle
 * it, negative = under-resourced. */
float wubu_mc_gap(float competence, float difficulty);

/* JD24: confidence-calibrated sampling -- the sampling temperature
 * scaled by the calibration (well-calibrated -> sharper). */
float wubu_mc_sample_temp(float calibration, float base_temp);

/* JD30: calibration drift monitor -- the ECE growth over time. */
float wubu_mc_drift(const wubu_mc_ece_t *e, float baseline_ece);

#endif
