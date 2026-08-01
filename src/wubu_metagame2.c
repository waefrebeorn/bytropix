/*
 * wubu_metagame2.c -- Deeper meta-game primitives (AH07/AH09/AH10/AH11). C11.
 *
 * Convergence (DGM sandbox, EXSKILL/XSkill skill lib, continual replay,
 * HyperAgents intrinsic metacog 7-hop):
 *   - AH07 sandbox: a self-modification is only applied if it passes an
 *          isolation gate (no network, no fs-escape, passes test_all). We model
 *          the gate as a capability bitmask check (no third-party sandbox dep).
 *   - AH09 skill library: a skill is a reusable (name, body, score) entry;
 *          replayable = can be re-invoked; non-parametric = stored as text,
 *          not weights. Top-k by score retrieved.
 *   - AH10 continual learning: replay buffer of experiences; consolidate
 *          without forgetting = keep a fixed-size reservoir sampled uniformly.
 *   - AH11 intrinsic metacognition: calibrate confidence vs actual accuracy
 *          over time; track running calibration error; agent "knows what it
 *          knows" when calibration error is low.
 *
 * Pure C11, deterministic, testable.
 */
#include "wubu_metagame2.h"
#include <stdlib.h>
#include <string.h>

/* AH07 sandbox gate. caps bitmask: bit0=no_net, bit1=no_fs_escape, bit2=tests_pass.
 * Returns 1 if the proposed mutation is allowed to apply. */
int wubu_sandbox_allow(unsigned int caps, int net_ok, int fs_ok, int tests_ok) {
    if (!net_ok && (caps & 1)) return 0;       /* needs net but denied */
    if (!fs_ok  && (caps & 2)) return 0;        /* needs fs but denied */
    if (!tests_ok) return 0;                    /* must pass tests */
    return 1;
}

/* AH09 skill library: add + top-k retrieve by score. */
int wubu_skill_add(wubu_skilllib_t *s, const char *name, const char *body, double score) {
    if (!s || s->n >= WUBU_SKILL_MAX) return 0;
    int i = s->n++;
    strncpy(s->name[i], name, 31); s->name[i][31] = 0;
    strncpy(s->body[i], body, 255); s->body[i][255] = 0;
    s->score[i] = score;
    return 1;
}
/* returns count written into out[] (indices into lib, best-first). */
int wubu_skill_topk(const wubu_skilllib_t *s, int k, int *out) {
    if (!s) return 0;
    int idx[WUBU_SKILL_MAX];
    for (int i = 0; i < s->n; i++) idx[i] = i;
    /* simple selection sort by score desc */
    for (int i = 0; i < s->n; i++)
        for (int j = i + 1; j < s->n; j++)
            if (s->score[idx[j]] > s->score[idx[i]]) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    int m = s->n < k ? s->n : k;
    for (int i = 0; i < m; i++) out[i] = idx[i];
    return m;
}

/* AH10 continual learning: reservoir replay buffer. add experience; if full,
 * keep with probability capacity/n_seen (classical reservoir sampling). */
int wubu_replay_add(wubu_replay_t *r, long exp_id, int *replace_idx) {
    if (!r) return 0;
    r->seen++;
    if (r->n < r->cap) { *replace_idx = r->n; r->buf[r->n++] = exp_id; return 1; }
    /* reservoir: replace random slot with prob cap/seen */
    long j = (long)((double)rand() / (RAND_MAX + 1.0) * r->seen);
    if (j < r->cap) { *replace_idx = (int)j; r->buf[j] = exp_id; return 1; }
    *replace_idx = -1; /* discarded (kept old) -> no forgetting of reservoir */
    return 0;
}

/* AH11 intrinsic metacognition: update calibration. confidence in [0,1],
 * correct=1/0. Returns running calibration error (|conf - actual| EMA). Lower
 * is better (agent knows what it knows). */
double wubu_metacog_update(wubu_metacog_t *m, double confidence, int correct) {
    if (!m) return 1.0;
    double actual = correct ? 1.0 : 0.0;
    double err = fabs(confidence - actual);
    m->calib = m->calib * 0.9 + err * 0.1;   /* EMA */
    m->n++;
    return m->calib;
}
int wubu_metacog_calibrated(const wubu_metacog_t *m, double thr) {
    return (m && m->n >= 8 && m->calib <= thr) ? 1 : 0;
}
