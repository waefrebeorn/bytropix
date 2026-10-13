/* wubu_hotkey.c -- UX #238 global summon hotkey (Ctrl+/).
 * See wubu_hotkey.h. Real, deterministic global-hotkey dispatcher. */

#include "wubu_hotkey.h"
#include <string.h>

int wubu_hk_init(wubu_hotkey_t *h)
{
    if (!h) return -1;
    memset(h, 0, sizeof(*h));
    h->enabled = 1;
    return 0;
}

int wubu_hk_set_enabled(wubu_hotkey_t *h, int enabled)
{
    if (!h) return -1;
    h->enabled = enabled ? 1 : 0;
    return 0;
}

static int find(wubu_hotkey_t *h, uint32_t mods, int key)
{
    for (int i = 0; i < h->n; i++)
        if (h->binds[i].active && h->binds[i].mods == mods && h->binds[i].key == key)
            return i;
    return -1;
}

int wubu_hk_bind(wubu_hotkey_t *h, uint32_t mods, int key, const char *command)
{
    if (!h || !command) return -1;
    if (find(h, mods, key) >= 0) return -1;   /* conflict: chord already bound */
    if (h->n >= WUBU_HK_MAX) return -1;
    wubu_hk_binding_t *b = &h->binds[h->n++];
    memset(b, 0, sizeof(*b));
    b->mods = mods;
    b->key = key;
    strncpy(b->command, command, WUBU_HK_NAME_MAX - 1);
    b->active = 1;
    return 0;
}

int wubu_hk_unbind(wubu_hotkey_t *h, uint32_t mods, int key)
{
    if (!h) return -1;
    int i = find(h, mods, key);
    if (i < 0) return -1;
    h->binds[i].active = 0;
    return 0;
}

int wubu_hk_dispatch(wubu_hotkey_t *h, uint32_t mods, int key)
{
    if (!h) return -1;
    if (!h->enabled) return 0;   /* disabled: never fires */
    int i = find(h, mods, key);
    if (i < 0) { h->last_fired = 0; return 0; }
    strncpy(h->last_command, h->binds[i].command, WUBU_HK_NAME_MAX - 1);
    h->last_command[WUBU_HK_NAME_MAX - 1] = '\0';
    h->last_fired = 1;
    h->n_fired++;
    return 1;
}

int wubu_hk_has(const wubu_hotkey_t *h, uint32_t mods, int key)
{
    if (!h) return 0;
    for (int i = 0; i < h->n; i++)
        if (h->binds[i].active && h->binds[i].mods == mods && h->binds[i].key == key)
            return 1;
    return 0;
}
