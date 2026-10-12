/*
 * wubu_provider_route.c -- local-vs-remote inference routing with offline
 * fallback (Track B5).
 *
 * The companion (and any WuBu agent) must ALWAYS work: with an API key it
 * reasons through a hosted provider; with no key it falls back to the local
 * WuBu-35M engine (gen_text) so the agent never goes dark. This module is
 * the routing decision: a PURE function (no I/O) that picks the backend
 * from availability, plus an env-reading wrapper that applies it to the
 * real environment.
 *
 * Policy (deny-to-dark is the failure we refuse): if a remote key is
 * present AND the caller allows remote, use remote; otherwise use local.
 * Local is the floor -- it is always an option when the binary + model are
 * present. If neither is available, the caller is told to surface an error,
 * never to silently pretend.
 *
 * C11, minimal includes. The pure decision is unit-testable with no env.
 */

#include "wubu_provider_route.h"
#include <stdlib.h>
#include <string.h>

/* Pure routing decision. Returns WUBU_ROUTE_REMOTE or WUBU_ROUTE_LOCAL. */
wubu_route_t wubu_provider_route_decide(int has_remote_key, int allow_remote,
                                        int has_local) {
    /* Remote only when a key exists, the caller permits it, and... we still
     * prefer local when remote is unavailable. */
    if (has_remote_key && allow_remote) return WUBU_ROUTE_REMOTE;
    /* No usable remote -> local is the floor. */
    if (has_local) return WUBU_ROUTE_LOCAL;
    /* Neither available: default to local (the caller surfaces the missing
     * binary/model error); we never silently route to a dead remote. */
    return WUBU_ROUTE_LOCAL;
}

/* Read the environment and decide. has_local is passed in (the caller knows
 * whether gen_text + the model are present; we can't stat that portably
 * here). Reads WUBU_API_KEY / OPENROUTER_API_KEY / NVIDIA_API_KEY. */
wubu_route_t wubu_provider_route_env(int has_local, int allow_remote) {
    const char *k = getenv("WUBU_API_KEY");
    if (!k || !k[0]) k = getenv("OPENROUTER_API_KEY");
    if (!k || !k[0]) k = getenv("NVIDIA_API_KEY");
    int has_remote_key = (k && k[0]) ? 1 : 0;
    return wubu_provider_route_decide(has_remote_key, allow_remote, has_local);
}

const char *wubu_provider_route_name(wubu_route_t r) {
    return r == WUBU_ROUTE_REMOTE ? "remote" : "local";
}
