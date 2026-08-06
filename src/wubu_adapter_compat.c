/* wubu_adapter_compat.c — compatibility shims for adapter versions
 *
 * Bridges between adapter API versions so old components can talk to
 * new ones (and vice versa). This is the "shim" layer: when the model
 * core is upgraded from attn_v1 to attn_v2, the v1 adapters still work
 * through a v1→v2 compatibility shim that translates the call.
 *
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_adapter.h"
#include "wubu_adapter_compat.h"
#include <string.h>

/* Register a compatibility shim: when an old adapter version is
 * requested but only the new version is available, the shim bridges. */
typedef struct {
    char old_name[64];
    char new_name[64];
    wubu_adapter_t *shim_adapter;  /* pre-built bridge adapter */
    int active;
} compat_mapping_t;

#define MAX_COMPAT 32
static compat_mapping_t g_maps[MAX_COMPAT];
static int g_n_maps = 0;

int wubu_adapter_register_shim(const char *old_name, const char *new_name) {
    if (!old_name || !new_name || g_n_maps >= MAX_COMPAT) return -1;
    wubu_adapter_t *new_adv = wubu_adapter_lookup(new_name);
    if (!new_adv) return -1;
    /* Build a bridge adapter that forwards to the new one */
    wubu_adapter_t *shim = (wubu_adapter_t *)calloc(1, sizeof(wubu_adapter_t));
    if (!shim) return -1;
    /* For now, the shim IS the new adapter (identity bridge).
     * In future, we'd wrap the ops vtable with translation logic. */
    shim->ops = new_adv->ops;
    g_maps[g_n_maps].shim_adapter = shim;
    strncpy(g_maps[g_n_maps].old_name, old_name, 63);
    g_maps[g_n_maps].old_name[63] = '\0';
    strncpy(g_maps[g_n_maps].new_name, new_name, 63);
    g_maps[g_n_maps].new_name[63] = '\0';
    g_maps[g_n_maps].active = 1;
    g_n_maps++;
    return 0;
}

/* Override lookup: if exact name not found, check compat maps.
 * This wraps wubu_adapter_lookup with shim fallback. */
wubu_adapter_t *wubu_adapter_lookup_compat(const char *name) {
    if (!name) return NULL;
    /* Try exact match first */
    wubu_adapter_t *adv = wubu_adapter_lookup(name);
    if (adv) return adv;
    /* Try compat shims */
    for (int i = 0; i < g_n_maps; i++) {
        if (g_maps[i].active &&
            strcmp(g_maps[i].old_name, name) == 0) {
            return g_maps[i].shim_adapter;
        }
    }
    return NULL;
}

int wubu_adapter_has_shim(const char *name) {
    if (!name) return 0;
    for (int i = 0; i < g_n_maps; i++) {
        if (g_maps[i].active &&
            strcmp(g_maps[i].old_name, name) == 0)
            return 1;
    }
    return 0;
}
