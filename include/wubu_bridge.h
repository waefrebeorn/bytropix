/*
 * wubu_bridge.h -- Cross-resource bridges (Theme JF, first batch). C11.
 * The Kevin-Bacon links between the new needs (JD metacog, JE Bonzi)
 * and the EXISTING modules. Each bridge is a small, testable function
 * that makes a link real (no print theater).
 */
#ifndef WUBU_BRIDGE_H
#define WUBU_BRIDGE_H

/* JF01: Bonzi mood memory -> Hopfield retrieval.
 * The mood (valence) is a 1-d pattern; retrieve the closest stored
 * mood pattern and return its associated response index. */
int wubu_br_mood_retrieve(const float *mood_patterns, int n_moods,
                          float valence, float *out_mood);

/* JF02: AGI confidence -> loopguard gate.
 * A calibrated confidence in [0,1]; the gate allows a high-risk action
 * only when confidence >= thresh AND the budget allows. */
int wubu_br_confidence_gate(float confidence, float thresh, int budget_left);

/* JF04: companion persona -> alignment guardrail.
 * The persona check: the response's safety score must clear the bar
 * before the persona may use it. */
int wubu_br_persona_guard(float safety_score, float bar);

/* JF06: self-assessment -> turn credit.
 * The credit earned by a turn = calibration * outcome (the calibrated
 * self-assessment times the actual outcome). */
float wubu_br_credit(float calibration, float outcome);

/* JF07: Bonzi chat pruning -> eviction.
 * Evict the oldest chat turn once the context exceeds the budget,
 * keeping the sink (first) turn. Returns the number evicted. */
int wubu_br_chat_prune(int turns, int budget);

/* JF10: metacog regulation -> policy.
 * The regulation action (0=stop, 1=retry, 2=delegate) selected from
 * the confidence + the remaining budget. */
int wubu_br_regulate(float confidence, int budget_left, float low, float mid);

/* JF15: mood prediction -> Gaussian process.
 * Predict the next mood from the current mood + a GP-style kernel
 * (exponential decay over time). */
float wubu_br_mood_predict(float current, int dt, float kernel_scale);

/* JF18: Bonzi forget -> unlearning.
 * The forget amount for a memory at age t under a forgetting curve;
 * returns the retained fraction. */
float wubu_br_forget_retain(int age, float halflife);

/* JF23: memory decay -> Hopfield decay schedule.
 * The effective weight of a companion memory at age t. */
float wubu_br_memory_weight(float base, int age, float halflife);

/* JF27: metacog + Hopfield -> self-knowledge patterns.
 * Encode a capability score into a self-knowledge pattern slot. */
int wubu_br_self_pattern(float capability, float *slot);

/* JF31: companion output verification.
 * Accept the output when the verifier score clears the bar. */
int wubu_br_verify_output(float verifier_score, float bar);

/* JF36: self-monitoring -> trajectory log.
 * Append a monitoring event (the monitoring log length returned). */
int wubu_br_monitor_log(int *log_len, int max_len);

/* JF45: Bonzi anomaly -> second-order monitor.
 * Flag when the mood delta exceeds the anomaly threshold. */
int wubu_br_mood_anomaly(float mood_delta, float thresh);

/* JF59: companion empathy -> DPO-style alignment.
 * The empathetic reward = beta * (empathy score gap between the chosen
 * and the rejected response). */
float wubu_br_empathy_reward(float emp_w, float emp_l, float beta);

/* JF74: self-assessment -> resource tier.
 * The resource tier (0=small,1=mid,2=large) from the calibrated
 * competence. */
int wubu_br_tier(float competence, float low, float high);

/* JF85: self-monitoring -> ECS component.
 * The monitoring state as a typed component snapshot. */
int wubu_br_monitor_component(float *component, int n, const float *vals);

/* JF100: close-rate ledger -> metagame.
 * Update the loop's close-rate: returns the new rolling rate. */
float wubu_br_close_rate(float prev_rate, int closed_this_batch, int batch_size);

#endif
