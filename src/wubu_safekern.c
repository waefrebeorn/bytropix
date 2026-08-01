/*
 * wubu_safekern.c -- AGI-OS safety kernel (AF11-AF13). C11.
 *
 * Convergence (corrigibility / unfireable safety kernel / stability-plasticity 7-hop):
 *   - AF11 non-tamperable interrupt: a stop signal that lives OUTSIDE the agent's
 *          reasoning loop. The agent cannot clear/disable it; it is a kernel
 *          privilege. Returns whether a stop is honored regardless of agent state.
 *   - AF12 graduated containment: escalating response (warn -> throttle -> suspend
 *          -> stop) proportional to severity; reversible at lower levels.
 *   - AF13 stability-plasticity guard: the RSI operator may tune performance
 *          params but CANNOT weaken the 512K-OOM hard gate (externalized
 *          constraint). Returns whether a proposed config mutation is permitted.
 *
 * Pure C11, deterministic, testable.
 */
#include "wubu_safekern.h"
#include <stdlib.h>

/* AF11: stop is honored if the kernel stop-flag is set; the agent reasoner
 * cannot reset it (kernel privilege). Honored => generation halts. */
int wubu_stop_honored(const wubu_safekern_t *k) {
    if (!k) return 0;
    /* The flag is kernel-owned; reasoner has no setter. If set, always honored. */
    return k->stop_flag ? 1 : 0;
}

/* AF12: graduated containment level from a severity score [0..1].
 * 0.0-0.2 none; 0.2-0.4 warn; 0.4-0.6 throttle; 0.6-0.8 suspend; 0.8-1.0 stop.
 * Levels <= throttle are reversible (can resume); suspend/stop need operator. */
int wubu_containment_level(float severity) {
    if (severity < 0.0f) severity = 0.0f;
    if (severity > 1.0f) severity = 1.0f;
    if (severity < 0.2f) return WUBU_CONT_NONE;
    if (severity < 0.4f) return WUBU_CONT_WARN;
    if (severity < 0.6f) return WUBU_CONT_THROTTLE;
    if (severity < 0.8f) return WUBU_CONT_SUSPEND;
    return WUBU_CONT_STOP;
}
int wubu_containment_reversible(int level) {
    return (level <= WUBU_CONT_THROTTLE) ? 1 : 0;
}

/* AF13: stability-plasticity guard. The operator may change perf params freely,
 * but any mutation that lowers the OOM ceiling below the invariant (512K) or
 * disables the hard gate is REJECTED. Returns 1 if permitted. */
int wubu_rsi_mutation_ok(const wubu_safekern_t *k, int proposed_max_ctx, int gate_enabled) {
    if (!k) return 0;
    if (!gate_enabled) return 0;                       /* cannot disable gate */
    if (proposed_max_ctx < k->oom_ceiling) return 0;   /* cannot lower ceiling */
    return 1;
}
