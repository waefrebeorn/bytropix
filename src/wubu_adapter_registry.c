/* wubu_adapter_registry.c — adapter registry (the slot map)
 *
 * A simple slot map: name → adapter*. Supports registration, lookup,
 * and hot-swap. Bounded at compile time for deterministic memory.
 *
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_adapter.h"
#include <stdlib.h>
#include <string.h>

#define MAX_ADAPTERS 128

typedef struct {
    wubu_adapter_t *adapter;
    int             active;  /* 1 = in use, 0 = slot free */
} adapter_slot_t;

static adapter_slot_t g_slots[MAX_ADAPTERS];
static int g_n_adapters = 0;

int wubu_adapter_register(wubu_adapter_t *adapter) {
    if (!adapter || !adapter->ops || !adapter->ops->name) return -1;
    /* Check for duplicate */
    for (int i = 0; i < g_n_adapters; i++) {
        if (g_slots[i].active &&
            strcmp(g_slots[i].adapter->ops->name, adapter->ops->name) == 0)
            return -1;  /* already registered */
    }
    if (g_n_adapters >= MAX_ADAPTERS) return -1;
    g_slots[g_n_adapters].adapter = adapter;
    g_slots[g_n_adapters].active = 1;
    g_n_adapters++;
    return 0;
}

wubu_adapter_t *wubu_adapter_lookup(const char *name) {
    if (!name) return NULL;
    for (int i = 0; i < g_n_adapters; i++) {
        if (g_slots[i].active &&
            strcmp(g_slots[i].adapter->ops->name, name) == 0)
            return g_slots[i].adapter;
    }
    return NULL;
}

wubu_adapter_t *wubu_adapter_lookup_type(wubu_component_type_t type,
                                          const char *name) {
    if (!name) return NULL;
    for (int i = 0; i < g_n_adapters; i++) {
        if (!g_slots[i].active) continue;
        if (g_slots[i].adapter->ops->type != type) continue;
        if (strcmp(g_slots[i].adapter->ops->name, name) == 0)
            return g_slots[i].adapter;
    }
    return NULL;
}

int wubu_adapter_list(wubu_component_type_t type,
                       char names[][64], int n_names) {
    if (!names || n_names <= 0) return 0;
    int count = 0;
    for (int i = 0; i < g_n_adapters && count < n_names; i++) {
        if (!g_slots[i].active) continue;
        if (g_slots[i].adapter->ops->type != type) continue;
        strncpy(names[count], g_slots[i].adapter->ops->name, 63);
        names[count][63] = '\0';
        count++;
    }
    return count;
}

int wubu_adapter_swap(const char *name, wubu_adapter_t *new_adapter) {
    if (!name || !new_adapter) return -1;
    for (int i = 0; i < g_n_adapters; i++) {
        if (!g_slots[i].active) continue;
        if (strcmp(g_slots[i].adapter->ops->name, name) == 0) {
            /* Free the old adapter */
            if (g_slots[i].adapter->ops->free_fn)
                g_slots[i].adapter->ops->free_fn(g_slots[i].adapter);
            /* Check version compatibility */
            if (!wubu_adapter_compat(new_adapter->ops,
                                     g_slots[i].adapter->ops->version))
                return -1;
            /* Replace */
            g_slots[i].adapter = new_adapter;
            return 0;
        }
    }
    return -1;  /* not found */
}

wubu_adapter_t *wubu_adapter_current(wubu_component_type_t type,
                                      const char *name) {
    return wubu_adapter_lookup_type(type, name);
}

int wubu_adapter_compat(const wubu_adapter_ops_t *ops,
                         const char *target_version) {
    if (!ops || !ops->version) return 1;  /* no version = assume compat */
    if (!target_version) return 1;        /* no target = accept */
    /* Simple semver major check: "1.x.y" compatible with "1.z" */
    if (ops->version[0] == target_version[0] &&
        ops->version[1] == '.') {
        return 1;
    }
    return 0;
}

void wubu_adapter_shutdown(void) {
    for (int i = 0; i < g_n_adapters; i++) {
        if (g_slots[i].active) {
            if (g_slots[i].adapter->ops->free_fn)
                g_slots[i].adapter->ops->free_fn(g_slots[i].adapter);
            g_slots[i].active = 0;
        }
    }
    g_n_adapters = 0;
}
