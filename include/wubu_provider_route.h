/*
 * wubu_provider_route.h -- local-vs-remote inference routing with offline
 * fallback (Track B5). The agent always works: remote with a key, local
 * (WuBu-35M gen_text) without. Pure decision + env wrapper.
 */
#ifndef WUBU_PROVIDER_ROUTE_H
#define WUBU_PROVIDER_ROUTE_H

typedef enum {
    WUBU_ROUTE_LOCAL  = 0,   /* the local WuBu-35M engine (gen_text) */
    WUBU_ROUTE_REMOTE = 1    /* a hosted provider (needs an API key) */
} wubu_route_t;

/* Pure routing decision (no I/O). Remote when a key exists AND the caller
 * permits it; otherwise local is the floor. Local is always the safe
 * default when remote is unavailable -- we never silently route to a dead
 * remote. */
wubu_route_t wubu_provider_route_decide(int has_remote_key, int allow_remote,
                                        int has_local);

/* Read the environment (WUBU_API_KEY / OPENROUTER_API_KEY / NVIDIA_API_KEY)
 * and decide. has_local is passed in (caller knows if gen_text + model are
 * present). */
wubu_route_t wubu_provider_route_env(int has_local, int allow_remote);

/* "remote" / "local" */
const char *wubu_provider_route_name(wubu_route_t r);

#endif /* WUBU_PROVIDER_ROUTE_H */
