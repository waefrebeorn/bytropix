/* wubu_hyprland.c -- OMARCHY #280 Hyprland Wayland compositor backend.
 * See wubu_hyprland.h. Real, deterministic state machine. */

#include "wubu_hyprland.h"
#include <string.h>

static void zerr(wubu_hyprland_t *hl, const char *msg) {
    if (hl && hl->state != WUBU_HL_STATE_DOWN && msg)
        strncpy(hl->last_error, msg, WUBU_HL_NAME_MAX - 1);
}

int wubu_hl_init(wubu_hyprland_t *hl) {
    if (!hl) return -1;
    memset(hl, 0, sizeof(*hl));
    hl->state = WUBU_HL_STATE_DOWN;
    /* default: one workspace, no outputs yet, handshake pending */
    return wubu_hl_add_workspace(hl, 1, "1");
}

int wubu_hl_run(wubu_hyprland_t *hl, const char *socket) {
    if (!hl || !socket) return -1;
    if (hl->state != WUBU_HL_STATE_DOWN) { zerr(hl, "run-not-down"); return -1; }
    /* INITING: would wl_display_connect(socket); here we model the state */
    hl->state = WUBU_HL_STATE_INITING;
    /* succeed connecting to the (modeled) socket -> RUNNING */
    if (socket[0] == '\0') { hl->state = WUBU_HL_STATE_DOWN; zerr(hl,"bad-socket"); return -1; }
    hl->state = WUBU_HL_STATE_RUNNING;
    return 0;
}

int wubu_hl_teardown(wubu_hyprland_t *hl) {
    if (!hl) return -1;
    if (hl->state != WUBU_HL_STATE_RUNNING && hl->state != WUBU_HL_STATE_INITING) { zerr(hl,"down-teardown"); return -1; }
    hl->state = WUBU_HL_STATE_TEARDOWN;
    hl->n_clients = hl->n_outputs = 0;   /* surfaces detached */
    hl->state = WUBU_HL_STATE_DOWN;
    hl->handshake_done = 0;
    return 0;
}

int wubu_hl_add_output(wubu_hyprland_t *hl, const char *name, int w, int h, int refresh_mhz) {
    if (!hl || !name) return -1;
    if (hl->state != WUBU_HL_STATE_RUNNING) { zerr(hl,"output-not-running"); return -1; }
    if (hl->n_outputs >= WUBU_HL_MAX_OUTPUTS) return -1;
    if (w <= 0 || h <= 0) return -1;
    wubu_hl_output_t *o = &hl->outputs[hl->n_outputs++];
    memset(o, 0, sizeof(*o));
    strncpy(o->name, name, WUBU_HL_NAME_MAX - 1);
    o->w = w; o->h = h; o->refresh = refresh_mhz; o->enabled = 1;
    return 0;
}

int wubu_hl_active_output(const wubu_hyprland_t *hl, int *idx, int *w, int *h) {
    if (!hl) return -1;
    int first = -1;
    for (int i = 0; i < hl->n_outputs; i++) {
        if (hl->outputs[i].enabled) { first = i; break; }
    }
    if (first < 0) return -1;
    if (idx) *idx = first;
    if (w)  *w  = hl->outputs[first].w;
    if (h)  *h  = hl->outputs[first].h;
    return 0;
}

int wubu_hl_add_workspace(wubu_hyprland_t *hl, int id, const char *name) {
    if (!hl || !name) return -1;
    if (hl->n_workspaces >= WUBU_HL_MAX_WORKSPACES) return -1;
    for (int i = 0; i < hl->n_workspaces; i++)
        if (hl->workspaces[i].id == id) return -1;  /* exists */
    wubu_hl_workspace_t *w = &hl->workspaces[hl->n_workspaces++];
    memset(w, 0, sizeof(*w));
    w->id = id;
    strncpy(w->name, name, WUBU_HL_NAME_MAX - 1);
    w->active = (id == 1) ? 1 : 0;   /* workspace 1 active by default */
    return 0;
}

int wubu_hl_workspace_of(const wubu_hyprland_t *hl, int ws_id) {
    if (!hl) return -1;
    for (int i = 0; i < hl->n_workspaces; i++)
        if (hl->workspaces[i].id == ws_id) return i;
    return -1;
}

int wubu_hl_add_client(wubu_hyprland_t *hl, const char *app_id, int pid) {
    if (!hl || !app_id) return -1;
    if (hl->state != WUBU_HL_STATE_RUNNING) { zerr((wubu_hyprland_t*)hl,"client-not-running"); return -1; }
    if (hl->n_clients >= WUBU_HL_MAX_CLIENTS) return -1;
    wubu_hl_client_t *c = &hl->clients[hl->n_clients++];
    memset(c, 0, sizeof(*c));
    c->id = hl->n_clients - 1;
    strncpy(c->app_id, app_id, WUBU_HL_NAME_MAX - 1);
    c->pid = pid;
    c->ws_id = 1;            /* default workspace */
    c->output_idx = -1;
    return c->id;
}

int wubu_hl_map_client(wubu_hyprland_t *hl, int cid, int ws_id, int output_idx,
                       int x, int y, int w, int h) {
    if (!hl || cid < 0 || cid >= hl->n_clients) return -1;
    if (wubu_hl_workspace_of(hl, ws_id) < 0) return -1;
    if (output_idx < 0 || output_idx >= hl->n_outputs) return -1;
    if (w <= 0 || h <= 0) return -1;
    wubu_hl_client_t *c = &hl->clients[cid];
    c->mapped = 1;
    c->ws_id = ws_id;
    c->output_idx = output_idx;
    c->wx = x; c->wy = y; c->ww = w; c->wh = h;
    /* count this client on its workspace */
    for (int i = 0; i < hl->n_workspaces; i++)
        if (hl->workspaces[i].id == ws_id) hl->workspaces[i].active_clients++;
    /* mark workspace active */
    for (int i = 0; i < hl->n_workspaces; i++)
        hl->workspaces[i].active = (hl->workspaces[i].id == ws_id);
    return 0;
}

int wubu_hl_unmap_client(wubu_hyprland_t *hl, int cid) {
    if (!hl || cid < 0 || cid >= hl->n_clients) return -1;
    wubu_hl_client_t *c = &hl->clients[cid];
    if (!c->mapped) return -1;
    c->mapped = 0;
    c->output_idx = -1;
    for (int i = 0; i < hl->n_workspaces; i++)
        if (hl->workspaces[i].id == c->ws_id) hl->workspaces[i].active_clients--;
    return 0;
}

int wubu_hl_client_at(const wubu_hyprland_t *hl, int x, int y) {
    if (!hl) return -1;
    /* topmost wins: iterate in reverse-add order (last mapped is on top) */
    for (int i = hl->n_clients - 1; i >= 0; i--) {
        const wubu_hl_client_t *c = &hl->clients[i];
        if (c->mapped && x >= c->wx && x < c->wx + c->ww &&
            y >= c->wy && y < c->wy + c->wh)
            return i;
    }
    return -1;
}

int wubu_hl_handshake(wubu_hyprland_t *hl) {
    if (!hl) return -1;
    if (hl->state != WUBU_HL_STATE_RUNNING) { zerr(hl,"hs-not-running"); return -1; }
    /* the compositor-WM handshake: advertise wlr-export-dmabuf + toplevel-mgmt */
    hl->handshake_done = 1;
    hl->dmabuf_implicit = 1;
    return 0;
}

int wubu_hl_set_dmabuf_implicit(wubu_hyprland_t *hl, int implicit) {
    if (!hl) return -1;
    hl->dmabuf_implicit = implicit ? 1 : 0;
    return 0;
}

int wubu_hl_state(const wubu_hyprland_t *hl) {
    if (!hl) return -1;
    return hl->state;
}

int wubu_hl_n_mapped(const wubu_hyprland_t *hl) {
    if (!hl) return -1;
    int n = 0;
    for (int i = 0; i < hl->n_clients; i++)
        if (hl->clients[i].mapped) n++;
    return n;
}
