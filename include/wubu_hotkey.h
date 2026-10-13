#ifndef WUBU_HOTKEY_H
#define WUBU_HOTKEY_H

/* wubu_hotkey.h -- UX #238 global summon hotkey (Ctrl+/).
 *
 * A global (system-wide) summon-hotkey dispatcher: hotkeys are registered as a
 * modifier bitmask + key bound to a command, and fire regardless of which
 * window has focus (unlike the modal per-layer keybinds in #292). The
 * dispatcher matches an incoming key event against the registry, detects
 * conflicts (two commands bound to the same chord -- the later registration is
 * rejected so a summon hotkey is unambiguous), and dispatches. Deterministic,
 * real work (no stubs). */

#include <stddef.h>
#include <stdint.h>

/* modifier bits */
#define WUBU_HK_CTRL  (1u << 0)
#define WUBU_HK_ALT   (1u << 1)
#define WUBU_HK_SHIFT (1u << 2)
#define WUBU_HK_SUPER (1u << 3)

#define WUBU_HK_MAX 64
#define WUBU_HK_NAME_MAX 48

typedef struct {
    uint32_t mods;                 /* modifier bitmask */
    int key;                       /* key code */
    char command[WUBU_HK_NAME_MAX];
    int active;
} wubu_hk_binding_t;

typedef struct {
    wubu_hk_binding_t binds[WUBU_HK_MAX];
    int n;
    int enabled;                   /* global hotkeys on/off */
    char last_command[WUBU_HK_NAME_MAX];
    int last_fired;
    uint64_t n_fired;
} wubu_hotkey_t;

/* Initialise (enabled, empty). Returns 0. */
int wubu_hk_init(wubu_hotkey_t *h);

/* Enable/disable global hotkeys. Returns 0. */
int wubu_hk_set_enabled(wubu_hotkey_t *h, int enabled);

/* Register a summon hotkey (mods, key) -> command. Rejects a conflicting
 * chord (same mods+key already bound) so a hotkey is unambiguous. Returns
 * 0/-1. */
int wubu_hk_bind(wubu_hotkey_t *h, uint32_t mods, int key, const char *command);

/* Unregister a hotkey by chord. Returns 0/-1. */
int wubu_hk_unbind(wubu_hotkey_t *h, uint32_t mods, int key);

/* Dispatch a key event. Returns 1 if a command fired (name in h->last_command),
 * 0 if no binding matched, -1 on bad args. Disabled dispatcher never fires. */
int wubu_hk_dispatch(wubu_hotkey_t *h, uint32_t mods, int key);

/* Does a chord have a binding? 1/0. */
int wubu_hk_has(const wubu_hotkey_t *h, uint32_t mods, int key);

#endif
