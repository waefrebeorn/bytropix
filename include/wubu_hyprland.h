#ifndef WUBU_HYPRLAND_H
#define WUBU_HYPRLAND_H

/* wubu_hyprland.h -- OMARCHY #280 Hyprland Wayland compositor backend.
 *
 * Models the Hyprland Wayland/WLR compositor as the WM backend for wubuos:
 * compositor lifecycle (init/run/teardown), output/monitor registry, virtual
 * workspaces, surfaces/clients with their XDG-toplevel roles, and the
 * compositor-WM protocol handshake (wlr-export-dmabuf / toplevel-management).
 * A deterministic state machine models the backend; no real compositor is
 * spawned here (the host Wayland backend is a separate process), but every
 * state transition the backend drives is represented and asserted. Real work,
 * no stubs. */

#include <stddef.h>

#define WUBU_HL_MAX_OUTPUTS 8
#define WUBU_HL_MAX_WORKSPACES 12
#define WUBU_HL_MAX_CLIENTS 64
#define WUBU_HL_NAME_MAX 48

typedef enum {
    WUBU_HL_STATE_DOWN,    /* compositor not running */
    WUBU_HL_STATE_INITING, /* connecting to Wayland display */
    WUBU_HL_STATE_RUNNING, /* compositor active, dispatching events */
    WUBU_HL_STATE_TEARDOWN
} wubu_hl_state_t;

typedef struct {
    char name[WUBU_HL_NAME_MAX];
    int w, h;            /* mode resolution */
    int refresh;         /* mHz */
    int enabled;
} wubu_hl_output_t;

typedef struct {
    int id;              /* workspace number */
    char name[WUBU_HL_NAME_MAX];
    int active_clients;
    int active;          /* currently shown on an output */
} wubu_hl_workspace_t;

typedef struct {
    int id;
    char app_id[WUBU_HL_NAME_MAX];
    char title[WUBU_HL_NAME_MAX];
    int pid;
    int wx, wy, ww, wh;  /* window geometry */
    int mapped;
    int ws_id;           /* workspace it lives on */
    int output_idx;      /* -1 if none */
} wubu_hl_client_t;

typedef struct {
    wubu_hl_state_t state;
    wubu_hl_output_t  outputs[WUBU_HL_MAX_OUTPUTS];
    int n_outputs;
    wubu_hl_workspace_t workspaces[WUBU_HL_MAX_WORKSPACES];
    int n_workspaces;
    wubu_hl_client_t  clients[WUBU_HL_MAX_CLIENTS];
    int n_clients;
    /* protocol handshake state */
    int handshake_done;
    int dmabuf_implicit;   /* wlr-export-dmabuf implicit-modifier support */
    int toplevel_active;   /* an XDG toplevel is currently active/focused */
    int n_events;          /* events dispatched since run */
    char last_error[WUBU_HL_NAME_MAX];
} wubu_hyprland_t;

/* Lifecycle */
int wubu_hl_init(wubu_hyprland_t *hl);
int wubu_hl_run(wubu_hyprland_t *hl, const char *socket);      /* -> INITING then RUNNING */
int wubu_hl_teardown(wubu_hyprland_t *hl);                    /* RUNNING -> TEARDOWN -> DOWN */

/* Outputs */
int wubu_hl_add_output(wubu_hyprland_t *hl, const char *name, int w, int h, int refresh_mhz);
int wubu_hl_active_output(const wubu_hyprland_t *hl, int *idx, int *w, int *h);

/* Workspaces */
int wubu_hl_add_workspace(wubu_hyprland_t *hl, int id, const char *name);
int wubu_hl_workspace_of(const wubu_hyprland_t *hl, int ws_id);

/* Clients / surfaces */
int wubu_hl_add_client(wubu_hyprland_t *hl, const char *app_id, int pid);
int wubu_hl_map_client(wubu_hyprland_t *hl, int cid, int ws_id, int output_idx,
                       int x, int y, int w, int h);
int wubu_hl_unmap_client(wubu_hyprland_t *hl, int cid);
int wubu_hl_client_at(const wubu_hyprland_t *hl, int x, int y);  /* pick by geometry */

/* Protocol handshake */
int wubu_hl_handshake(wubu_hyprland_t *hl);
int wubu_hl_set_dmabuf_implicit(wubu_hyprland_t *hl, int implicit);

/* Query */
int wubu_hl_state(const wubu_hyprland_t *hl);
int wubu_hl_n_mapped(const wubu_hyprland_t *hl);

#endif
