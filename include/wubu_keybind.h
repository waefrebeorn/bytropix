#ifndef WUBU_KEYBIND_H
#define WUBU_KEYBIND_H

/* wubu_keybind.h -- OMARCHY #292 Omarchy-style keybind-driven UX.
 *
 * A keyboard-first command dispatcher in the Omarchy style: key sequences
 * (chords) are bound to commands, organised into modal layers. A sequence may
 * start with a leader/prefix key (e.g. SUPER) that arms a sub-map; the
 * dispatcher walks the chord key-by-key, tracking partial matches, and fires
 * the bound command on an exact match. Unknown keys or timeouts abort the
 * pending sequence (fail-safe). Deterministic, real work (no stubs). */

#include <stddef.h>
#include <stdint.h>

#define WUBU_KB_MAX_BINDINGS 128
#define WUBU_KB_MAX_LAYERS 8
#define WUBU_KB_KEY_LEN 8          /* max keys per chord */
#define WUBU_KB_NAME_MAX 48

typedef struct {
    int keys[WUBU_KB_KEY_LEN];     /* chord sequence, 0-terminated */
    int n_keys;
    char layer[WUBU_KB_NAME_MAX];  /* layer this binding is active in */
    char command[WUBU_KB_NAME_MAX];/* command fired on exact match */
} wubu_kb_binding_t;

typedef struct {
    wubu_kb_binding_t binds[WUBU_KB_MAX_BINDINGS];
    int n_binds;
    char layers[WUBU_KB_MAX_LAYERS][WUBU_KB_NAME_MAX];
    int n_layers;
    char active_layer[WUBU_KB_NAME_MAX];
    /* pending partial-match state */
    int pending[WUBU_KB_KEY_LEN];
    int n_pending;
    int armed;                     /* 1 while a partial chord is in progress */
    char last_command[WUBU_KB_NAME_MAX];
    int last_fired;                /* 1 if last_key fired a command */
} wubu_keybind_t;

/* Initialise (default layer "root", empty). Returns 0. */
int wubu_kb_init(wubu_keybind_t *k);

/* Register a layer name. Returns 0/-1 (dup or full). */
int wubu_kb_add_layer(wubu_keybind_t *k, const char *layer);

/* Switch the active layer. Returns 0/-1 (unknown layer). Aborts any pending
 * chord. */
int wubu_kb_set_layer(wubu_keybind_t *k, const char *layer);

/* Bind a chord (keys[0..n-1]) in `layer` to `command`. Returns 0/-1. */
int wubu_kb_bind(wubu_keybind_t *k, const char *layer,
                 const int *keys, int n_keys, const char *command);

/* Feed one key. Advances a partial match or fires a command on exact match.
 * Returns 1 if a command fired (its name in k->last_command), 0 if the key was
 * consumed as part of a pending chord, -1 if the key aborted the sequence
 * (no binding starts with it) or on bad args. */
int wubu_kb_key(wubu_keybind_t *k, int key);

/* Abort any pending chord (e.g. on timeout). Returns 0. */
int wubu_kb_reset(wubu_keybind_t *k);

/* Is a chord currently in progress? 1/0. */
int wubu_kb_pending(const wubu_keybind_t *k);

#endif
