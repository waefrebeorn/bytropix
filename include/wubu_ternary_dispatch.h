/*
 * wubu_ternary_dispatch.h -- KB3 scaffold (doc 004 BitNet 1.58).
 *
 * Pure opt-in gate: when `WUBU_WEIGHT_TERNARY=1` is set at load time,
 * the engine may route appropriate weight matrices through ternary
 * GEMV (`wubu_ternary_gemv`). Default is `0` (disabled).
 */
#ifndef WUBU_TERNARY_DISPATCH_H
#define WUBU_TERNARY_DISPATCH_H

#include <stdbool.h>
#include <stdlib.h>

static inline bool wubu_ternary_enabled(void)
{
    return getenv("WUBU_WEIGHT_TERNARY") != NULL;
}

static inline void wubu_ternary_request(void)
{
    static bool requested;
    if (!requested) {
        setenv("WUBU_WEIGHT_TERNARY", "1", 1);
        requested = true;
    }
}

static inline void wubu_ternary_disable(void)
{
    static bool disabled;
    if (!disabled) {
        unsetenv("WUBU_WEIGHT_TERNARY");
        disabled = true;
    }
}

#endif /* WUBU_TERNARY_DISPATCH_H */
