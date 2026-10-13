/* wubu_steam_workshop.c -- Steam Workshop / mod manager.
 * See wubu_steam_workshop.h. Real, deterministic mod manager. */

#include "wubu_steam_workshop.h"
#include <string.h>

int wubu_ws_init(wubu_steam_workshop_t *w) {
    if (!w) return -1;
    memset(w, 0, sizeof(*w));
    w->next_id = 1;
    return 0;
}

int wubu_ws_find(const wubu_steam_workshop_t *w, uint64_t id) {
    if (!w) return -1;
    for (int i = 0; i < w->n_mods; i++)
        if (w->mods[i].id == id) return i;
    return -1;
}

static int find_in_order(const wubu_steam_workshop_t *w, uint64_t id) {
    for (int i = 0; i < w->n_load_order; i++)
        if (w->load_order[i] == id) return i;
    return -1;
}

uint64_t wubu_ws_subscribe(wubu_steam_workshop_t *w, const char *name,
                           const char *author) {
    if (!w || !name || !author) return (uint64_t)-1;
    if (w->n_mods >= WUBU_WS_MAX_MODS) return (uint64_t)-1;
    wubu_ws_mod_t *m = &w->mods[w->n_mods];
    memset(m, 0, sizeof(*m));
    m->id = w->next_id++;
    strncpy(m->name, name, WUBU_WS_NAME_MAX - 1);
    strncpy(m->author, author, WUBU_WS_NAME_MAX - 1);
    m->dl = WUBU_WS_SUBSCRIBED;
    m->enabled = 1;
    m->version = 1;
    w->n_mods++;
    return m->id;
}

int wubu_ws_unsubscribe(wubu_steam_workshop_t *w, uint64_t id) {
    if (!w) return -1;
    int i = wubu_ws_find(w, id);
    if (i < 0) return -1;
    /* remove from load order */
    int lo = find_in_order(w, id);
    if (lo >= 0) {
        for (int k = lo; k < w->n_load_order - 1; k++)
            w->load_order[k] = w->load_order[k + 1];
        w->n_load_order--;
    }
    /* remove the mod (swap with last) */
    w->mods[i] = w->mods[w->n_mods - 1];
    w->n_mods--;
    return 0;
}

int wubu_ws_add_dep(wubu_steam_workshop_t *w, uint64_t id, uint64_t dep_id) {
    if (!w) return -1;
    int i = wubu_ws_find(w, id);
    if (i < 0) return -1;
    if (wubu_ws_find(w, dep_id) < 0) return -1;   /* dep must exist */
    if (dep_id == id) return -1;                  /* no self-dep */
    wubu_ws_mod_t *m = &w->mods[i];
    for (int k = 0; k < m->n_deps; k++)
        if (m->deps[k] == dep_id) return 0;        /* already a dep */
    if (m->n_deps >= WUBU_WS_MAX_DEPS) return -1;
    m->deps[m->n_deps++] = dep_id;
    return 0;
}

int wubu_ws_add_file(wubu_steam_workshop_t *w, uint64_t id, const char *file) {
    if (!w || !file) return -1;
    int i = wubu_ws_find(w, id);
    if (i < 0) return -1;
    wubu_ws_mod_t *m = &w->mods[i];
    if (m->n_files >= WUBU_WS_MAX_FILES) return -1;
    strncpy(m->files[m->n_files], file, WUBU_WS_FILE_MAX - 1);
    m->files[m->n_files][WUBU_WS_FILE_MAX - 1] = '\0';
    m->n_files++;
    return 0;
}

int wubu_ws_download_start(wubu_steam_workshop_t *w, uint64_t id) {
    int i = wubu_ws_find(w, id);
    if (i < 0) return -1;
    if (w->mods[i].dl != WUBU_WS_SUBSCRIBED && w->mods[i].dl != WUBU_WS_FAILED)
        return -1;
    w->mods[i].dl = WUBU_WS_DOWNLOADING;
    return 0;
}

int wubu_ws_download_done(wubu_steam_workshop_t *w, uint64_t id) {
    int i = wubu_ws_find(w, id);
    if (i < 0) return -1;
    if (w->mods[i].dl != WUBU_WS_DOWNLOADING) return -1;
    w->mods[i].dl = WUBU_WS_DOWNLOADED;
    return 0;
}

int wubu_ws_download_fail(wubu_steam_workshop_t *w, uint64_t id) {
    int i = wubu_ws_find(w, id);
    if (i < 0) return -1;
    if (w->mods[i].dl != WUBU_WS_DOWNLOADING) return -1;
    w->mods[i].dl = WUBU_WS_FAILED;
    return 0;
}

int wubu_ws_set_enabled(wubu_steam_workshop_t *w, uint64_t id, int enabled) {
    int i = wubu_ws_find(w, id);
    if (i < 0) return -1;
    w->mods[i].enabled = enabled ? 1 : 0;
    return 0;
}

int wubu_ws_load(wubu_steam_workshop_t *w, uint64_t id) {
    if (!w) return -1;
    int i = wubu_ws_find(w, id);
    if (i < 0) return -1;
    if (w->mods[i].dl != WUBU_WS_DOWNLOADED) return -1;   /* must be downloaded */
    if (!w->mods[i].enabled) return -1;                   /* must be enabled */
    if (find_in_order(w, id) >= 0) return 0;              /* already loaded */
    if (w->n_load_order >= WUBU_WS_MAX_MODS) return -1;
    w->load_order[w->n_load_order++] = id;
    return 0;
}

int wubu_ws_unload(wubu_steam_workshop_t *w, uint64_t id) {
    if (!w) return -1;
    int lo = find_in_order(w, id);
    if (lo < 0) return -1;
    for (int k = lo; k < w->n_load_order - 1; k++)
        w->load_order[k] = w->load_order[k + 1];
    w->n_load_order--;
    return 0;
}

int wubu_ws_reorder(wubu_steam_workshop_t *w, uint64_t id, int delta) {
    if (!w) return -1;
    int lo = find_in_order(w, id);
    if (lo < 0) return -1;
    int to = lo + delta;
    if (to < 0 || to >= w->n_load_order) return -1;
    uint64_t tmp = w->load_order[lo];
    w->load_order[lo] = w->load_order[to];
    w->load_order[to] = tmp;
    return 0;
}

int wubu_ws_count_conflicts(wubu_steam_workshop_t *w) {
    if (!w) return -1;
    int conflicts = 0;
    /* for each pair of loaded+enabled mods, count shared files */
    for (int a = 0; a < w->n_load_order; a++) {
        int ia = wubu_ws_find(w, w->load_order[a]);
        if (ia < 0 || !w->mods[ia].enabled) continue;
        for (int b = a + 1; b < w->n_load_order; b++) {
            int ib = wubu_ws_find(w, w->load_order[b]);
            if (ib < 0 || !w->mods[ib].enabled) continue;
            for (int fa = 0; fa < w->mods[ia].n_files; fa++)
                for (int fb = 0; fb < w->mods[ib].n_files; fb++)
                    if (strcmp(w->mods[ia].files[fa], w->mods[ib].files[fb]) == 0)
                        conflicts++;
        }
    }
    w->n_conflicts = conflicts;
    return conflicts;
}

wubu_ws_dlstate_t wubu_ws_dl_state(const wubu_steam_workshop_t *w, uint64_t id) {
    int i = wubu_ws_find(w, id);
    if (i < 0) return WUBU_WS_NONE;
    return w->mods[i].dl;
}

int wubu_ws_n_loaded(const wubu_steam_workshop_t *w) {
    if (!w) return -1;
    return w->n_load_order;
}
