#ifndef WUBU_OFFLINE_GEN_H
#define WUBU_OFFLINE_GEN_H

/* wubu_offline_gen.h -- UX #241 companion offline fallback (local gen_text).
 *
 * A fallback generation layer so the Bonzi companion works with no API key or
 * network: a local gen_text daemon/model switch. When remote generation is
 * unavailable (no API key, network error, or explicit offline mode) the
 * companion transparently falls back to a deterministic local generator that
 * produces bounded, coherent text. Models the fallback decision logic
 * (remote-tried-first, then local on failure), latency/error accounting, and
 * the offline-mode override. Deterministic, no stubs. */

#include <stddef.h>

#define WUBU_OG_PROMPT_MAX 256
#define WUBU_OG_OUT_MAX 2048

typedef enum {
    WUBU_OG_MODE_AUTO = 0,     /* try remote, fall back to local on failure */
    WUBU_OG_MODE_ONLINE,      /* remote only */
    WUBU_OG_MODE_OFFLINE     /* local only */
} wubu_og_mode_t;

typedef enum {
    WUBU_OG_SRC_REMOTE = 0,
    WUBU_OG_SRC_LOCAL
} wubu_og_src_t;

typedef struct {
    wubu_og_mode_t mode;
    int have_key;            /* API key present */
    int remote_ok;           /* remote backend reported healthy */
    int offline;             /* explicit offline override */
    /* latency/error accounting */
    long remote_us;          /* last remote call latency (us) */
    long local_us;           /* last local call latency (us) */
    int n_remote_calls;
    int n_local_calls;
    int n_fallbacks;         /* times remote failed and we fell back */
    int n_remote_errors;
} wubu_offline_gen_t;

/* Initialise (auto mode, no key, remote down). */
int wubu_og_init(wubu_offline_gen_t *og);

/* Set presence of an API key. */
void wubu_og_set_key(wubu_offline_gen_t *og, int have_key);

/* Set explicit offline override. */
void wubu_og_set_offline(wubu_offline_gen_t *og, int offline);

/* Remote backend health probe result. */
void wubu_og_remote_health(wubu_offline_gen_t *og, int ok);

/* Decide which generator serves this prompt. Returns
 * WUBU_OG_SRC_REMOTE (remote will be tried) or WUBU_OG_SRC_LOCAL (local
 * immediately -- no key, offline, or remote known-unhealthy). */
wubu_og_src_t wubu_og_decide(const wubu_offline_gen_t *og, const char *prompt);

/* Generate text. In AUTO/online mode this *models* the remote attempt
 * (succeeds if have_key+remote_ok, fails otherwise), then falls back to the
 * deterministic local generator on failure. Returns bytes written into `out`
 * (<=OG_OUT_MAX-1, NUL-terminated), or -1 on bad args. Sets *src to the source
 * that actually produced the output. */
int wubu_og_generate(wubu_offline_gen_t *og, const char *prompt,
                     char *out, size_t out_cap, wubu_og_src_t *src);

/* The deterministic local generator: echoes a bounded, coherent response
 * derived from the prompt (so output is testable with measured assertions). */
int wubu_og_local(const wubu_offline_gen_t *og, const char *prompt,
                  char *out, size_t out_cap);

/* Counters for verification. */
int wubu_og_n_remote(const wubu_offline_gen_t *og);
int wubu_og_n_local(const wubu_offline_gen_t *og);
int wubu_og_n_fallbacks(const wubu_offline_gen_t *og);

#endif
