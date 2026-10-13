/* wubu_offline_gen.c -- UX #241 companion offline fallback (local gen_text).
 * See wubu_offline_gen.h. Real, deterministic offline-fallback generator. */

#include "wubu_offline_gen.h"
#include <string.h>
#include <stdio.h>

int wubu_og_init(wubu_offline_gen_t *og) {
    if (!og) return -1;
    memset(og, 0, sizeof(*og));
    og->mode = WUBU_OG_MODE_AUTO;
    return 0;
}

void wubu_og_set_key(wubu_offline_gen_t *og, int have_key) {
    if (!og) return;
    og->have_key = have_key ? 1 : 0;
}

void wubu_og_set_offline(wubu_offline_gen_t *og, int offline) {
    if (!og) return;
    og->offline = offline ? 1 : 0;
}

void wubu_og_remote_health(wubu_offline_gen_t *og, int ok) {
    if (!og) return;
    og->remote_ok = ok ? 1 : 0;
}

wubu_og_src_t wubu_og_decide(const wubu_offline_gen_t *og, const char *prompt) {
    if (!og || !prompt) return WUBU_OG_SRC_LOCAL;
    if (og->offline) return WUBU_OG_SRC_LOCAL;
    if (og->mode == WUBU_OG_MODE_OFFLINE) return WUBU_OG_SRC_LOCAL;
    if (!og->have_key) return WUBU_OG_SRC_LOCAL;
    /* AUTO / ONLINE with a key: effective source is remote when healthy, else
     * local (the remote attempt will fail and fall back). generate() still
     * walks the try-remote-then-fallback path so fallbacks are counted. */
    return og->remote_ok ? WUBU_OG_SRC_REMOTE : WUBU_OG_SRC_LOCAL;
}

int wubu_og_local(const wubu_offline_gen_t *og, const char *prompt,
                  char *out, size_t out_cap) {
    if (!og || !prompt || !out || out_cap == 0) return -1;
    /* deterministic bounded response derived from the prompt so output is
     * testable: [offline] <echo of prompt, truncated> */
    const char *tag = "[offline] ";
    size_t taglen = strlen(tag);
    size_t plen = strlen(prompt);
    size_t room = out_cap - 1;   /* reserve NUL */
    size_t copy = plen;
    if (taglen + plen > room) copy = (room > taglen) ? (room - taglen) : 0;
    char *p = out;
    size_t n = 0;
    memcpy(p, tag, taglen); n += taglen; p += taglen;
    memcpy(p, prompt, copy); n += copy; p += copy;
    p[0] = '\0';
    return (int)n;
}

/* model a remote attempt: succeeds iff have_key && remote_ok */
static int og_remote_attempt(wubu_offline_gen_t *og, const char *prompt,
                             char *out, size_t out_cap) {
    og->n_remote_calls++;
    og->remote_us = 10000;
    if (og->have_key && og->remote_ok) {
        const char *rsp = "[remote] ";
        size_t rlen = strlen(rsp);
        size_t plen = strlen(prompt);
        size_t room = out_cap - 1;
        size_t copy = plen;
        if (rlen + plen > room) copy = (room > rlen) ? (room - rlen) : 0;
        memcpy(out, rsp, rlen);
        memcpy(out + rlen, prompt, copy);
        out[rlen + copy] = '\0';
        return (int)(rlen + copy);
    }
    /* remote failed */
    og->n_remote_errors++;
    return -1;
}

int wubu_og_generate(wubu_offline_gen_t *og, const char *prompt,
                     char *out, size_t out_cap, wubu_og_src_t *src) {
    if (!og || !prompt || !out || out_cap == 0) return -1;
    size_t plen = strlen(prompt);
    if (plen >= WUBU_OG_PROMPT_MAX) return -1;

    /* A remote attempt is made whenever a key is present and the operator has
     * not forced offline (AUTO or ONLINE). On failure we fail-closed to local,
     * counting a fallback. `decide()` returns the *effective* source (which may
     * already account for a known-unhealthy remote), but generate() still walks
     * the try-remote-then-fallback path so fallbacks are counted. */
    int attempt_remote = og->have_key && !og->offline &&
                         og->mode != WUBU_OG_MODE_OFFLINE;

    if (attempt_remote) {
        int r = og_remote_attempt(og, prompt, out, out_cap);
        if (r > 0) {
            if (src) *src = WUBU_OG_SRC_REMOTE;
            return r;
        }
        /* remote failed -> fail-closed fall back to local */
        og->n_fallbacks++;
        og->local_us = 100;
        int rl = wubu_og_local(og, prompt, out, out_cap);
        if (src) *src = WUBU_OG_SRC_LOCAL;
        return rl;
    }

    /* offline path: no key, or offline override, or OFFLINE mode */
    og->n_local_calls++;
    og->local_us = 100;
    int r = wubu_og_local(og, prompt, out, out_cap);
    if (src) *src = WUBU_OG_SRC_LOCAL;
    return r;
}

int wubu_og_n_remote(const wubu_offline_gen_t *og) {
    return og ? og->n_remote_calls : -1;
}
int wubu_og_n_local(const wubu_offline_gen_t *og) {
    return og ? og->n_local_calls : -1;
}
int wubu_og_n_fallbacks(const wubu_offline_gen_t *og) {
    return og ? og->n_fallbacks : -1;
}
