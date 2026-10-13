/* wubu_keybind.c -- OMARCHY #292 Omarchy-style keybind-driven UX.
 * See wubu_keybind.h. Real, deterministic chord dispatcher. */

#include "wubu_keybind.h"
#include <string.h>

int wubu_kb_init(wubu_keybind_t *k)
{
    if (!k) return -1;
    memset(k, 0, sizeof(*k));
    strncpy(k->active_layer, "root", WUBU_KB_NAME_MAX - 1);
    strncpy(k->layers[k->n_layers++], "root", WUBU_KB_NAME_MAX - 1);
    return 0;
}

int wubu_kb_add_layer(wubu_keybind_t *k, const char *layer)
{
    if (!k || !layer) return -1;
    for (int i = 0; i < k->n_layers; i++)
        if (strcmp(k->layers[i], layer) == 0) return -1;   /* dup */
    if (k->n_layers >= WUBU_KB_MAX_LAYERS) return -1;
    strncpy(k->layers[k->n_layers++], layer, WUBU_KB_NAME_MAX - 1);
    return 0;
}

int wubu_kb_set_layer(wubu_keybind_t *k, const char *layer)
{
    if (!k || !layer) return -1;
    for (int i = 0; i < k->n_layers; i++)
        if (strcmp(k->layers[i], layer) == 0) {
            strncpy(k->active_layer, layer, WUBU_KB_NAME_MAX - 1);
            k->active_layer[WUBU_KB_NAME_MAX - 1] = '\0';
            wubu_kb_reset(k);
            return 0;
        }
    return -1;
}

int wubu_kb_bind(wubu_keybind_t *k, const char *layer,
                 const int *keys, int n_keys, const char *command)
{
    if (!k || !layer || !keys || !command) return -1;
    if (n_keys <= 0 || n_keys > WUBU_KB_KEY_LEN) return -1;
    if (k->n_binds >= WUBU_KB_MAX_BINDINGS) return -1;
    wubu_kb_binding_t *b = &k->binds[k->n_binds++];
    for (int i = 0; i < n_keys; i++) b->keys[i] = keys[i];
    b->n_keys = n_keys;
    strncpy(b->layer, layer, WUBU_KB_NAME_MAX - 1);
    b->layer[WUBU_KB_NAME_MAX - 1] = '\0';
    strncpy(b->command, command, WUBU_KB_NAME_MAX - 1);
    b->command[WUBU_KB_NAME_MAX - 1] = '\0';
    return 0;
}

int wubu_kb_reset(wubu_keybind_t *k)
{
    if (!k) return -1;
    k->n_pending = 0;
    k->armed = 0;
    return 0;
}

int wubu_kb_pending(const wubu_keybind_t *k)
{
    return k ? k->armed : 0;
}

/* Does any binding in the active layer start with pending+key? */
static int has_prefix(wubu_keybind_t *k, const int *seq, int n)
{
    for (int i = 0; i < k->n_binds; i++) {
        wubu_kb_binding_t *b = &k->binds[i];
        if (strcmp(b->layer, k->active_layer) != 0) continue;
        if (b->n_keys < n) continue;
        int ok = 1;
        for (int j = 0; j < n; j++) if (b->keys[j] != seq[j]) { ok = 0; break; }
        if (ok) return 1;
    }
    return 0;
}

/* Exact match in the active layer? returns binding index or -1. */
static int exact_match(wubu_keybind_t *k, const int *seq, int n)
{
    for (int i = 0; i < k->n_binds; i++) {
        wubu_kb_binding_t *b = &k->binds[i];
        if (strcmp(b->layer, k->active_layer) != 0) continue;
        if (b->n_keys != n) continue;
        int ok = 1;
        for (int j = 0; j < n; j++) if (b->keys[j] != seq[j]) { ok = 0; break; }
        if (ok) return i;
    }
    return -1;
}

int wubu_kb_key(wubu_keybind_t *k, int key)
{
    if (!k) return -1;
    /* candidate sequence = pending + this key */
    int cand[WUBU_KB_KEY_LEN];
    int n = 0;
    for (int i = 0; i < k->n_pending; i++) cand[n++] = k->pending[i];
    if (n >= WUBU_KB_KEY_LEN) { wubu_kb_reset(k); return -1; }
    cand[n++] = key;

    /* exact match fires the command */
    int em = exact_match(k, cand, n);
    if (em >= 0) {
        strncpy(k->last_command, k->binds[em].command, WUBU_KB_NAME_MAX - 1);
        k->last_command[WUBU_KB_NAME_MAX - 1] = '\0';
        k->last_fired = 1;
        wubu_kb_reset(k);
        return 1;
    }
    /* otherwise it must be a prefix of some longer chord */
    if (has_prefix(k, cand, n)) {
        for (int i = 0; i < n; i++) k->pending[i] = cand[i];
        k->n_pending = n;
        k->armed = 1;
        k->last_fired = 0;
        return 0;
    }
    /* no binding matches: abort (fail-safe) */
    wubu_kb_reset(k);
    k->last_fired = 0;
    return -1;
}
