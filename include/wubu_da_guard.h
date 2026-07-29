#ifndef WUBU_DA_GUARD_H
#define WUBU_DA_GUARD_H

#include <stdlib.h>

/*
 * wubu_da_guard.h — DA-1/DA-2 fail-closed model-load gate.
 *
 * WuBuOS realm publishes its trace-schema major via the WUBU_KERNEL_SCHEMA
 * environment variable at realm start. wubu_model_adapter checks this before
 * loading any weights: if the kernel schema doesn't match our compile-time
 * constant, the load refuses (DA-2: no silent telemetry gap, fail-closed).
 *
 * This is a zero-dependency check — no realm struct needed, works on WSL-hosted
 * and bare-metal. The env var is set by wubu_realm_start() in WuBuOS.
 */
#define WUBU_TRACE_SCHEMA_MAJOR 1

static inline int wubu_da_check_kernel_schema(void) {
    const char *env = getenv("WUBU_KERNEL_SCHEMA");
    if (!env || env[0] == 0) {
        /* No realm — standalone wubuwizard (e.g. gen_text). Allow. */
        return 0;
    }
    int kernel_major = atoi(env);
    if (kernel_major != WUBU_TRACE_SCHEMA_MAJOR) {
        /* DA-2 fail-closed: schema mismatch means the kernel was tampered
         * or is a different version. Refuse to load weights. */
        return -1;
    }
    return 0;
}

#endif /* WUBU_DA_GUARD_H */
