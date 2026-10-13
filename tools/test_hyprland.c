/* test_hyprland.c -- OMARCHY #280 tests: Hyprland Wayland compositor backend. */
#include "wubu_hyprland.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_hyprland_t hl;

    /* init: DOWN state, default workspace 1 active */
    CHECK(wubu_hl_init(&hl) == 0);
    CHECK(wubu_hl_state(&hl) == WUBU_HL_STATE_DOWN);
    CHECK(hl.n_outputs == 0);

    /* run on a socket -> RUNNING */
    CHECK(wubu_hl_run(&hl, "wayland-1") == 0);
    CHECK(wubu_hl_state(&hl) == WUBU_HL_STATE_RUNNING);
    /* bad socket rejected */
    CHECK(wubu_hl_run(&hl, "") == -1);
    CHECK(wubu_hl_state(&hl) == WUBU_HL_STATE_RUNNING);  /* unchanged */

    /* empty socket rejected from DOWN */
    CHECK(wubu_hl_run(&hl, "") == -1);

    /* outputs: two monitors */
    CHECK(wubu_hl_add_output(&hl, "eDP-1", 1920, 1080, 60000) == 0);
    CHECK(wubu_hl_add_output(&hl, "HDMI-A-1", 2560, 1440, 144000) == 0);
    CHECK(hl.n_outputs == 2);
    CHECK(strcmp(hl.outputs[0].name, "eDP-1") == 0);
    CHECK(hl.outputs[1].w == 2560 && hl.outputs[1].h == 1440);
    /* bad output */
    CHECK(wubu_hl_add_output(&hl, "bad", 0, 0, 60000) == -1);
    CHECK(wubu_hl_add_output(&hl, "bad2", -1, 1080, 60000) == -1);

    /* active output = first enabled */
    int idx, w, h;
    CHECK(wubu_hl_active_output(&hl, &idx, &w, &h) == 0);
    CHECK(idx == 0 && w == 1920 && h == 1080);

    /* add a second workspace (3 total incl default 1) */
    CHECK(wubu_hl_add_workspace(&hl, 2, "2") == 0);
    CHECK(wubu_hl_add_workspace(&hl, 9, "9") == 0);
    CHECK(wubu_hl_workspace_of(&hl, 1) == 0);
    CHECK(wubu_hl_workspace_of(&hl, 9) == 2);
    CHECK(wubu_hl_workspace_of(&hl, 5) == -1);
    /* duplicate workspace rejected */
    CHECK(wubu_hl_add_workspace(&hl, 1, "dup") == -1);
    CHECK(hl.n_workspaces == 3);
    /* workspace 1 active by default */
    CHECK(hl.workspaces[0].active == 1);

    /* clients */
    int term = wubu_hl_add_client(&hl, "terminal", 1001); CHECK(term == 0);
    int browser = wubu_hl_add_client(&hl, "firefox", 1002); CHECK(browser == 1);
    int editor = wubu_hl_add_client(&hl, "neovim", 1003); CHECK(editor == 2);
    CHECK(hl.n_clients == 3);
    CHECK(hl.clients[0].pid == 1001);
    CHECK(hl.clients[browser].ws_id == 1);

    /* map term on workspace 1, output 0 */
    CHECK(wubu_hl_map_client(&hl, term, 1, 0, 0, 0, 800, 600) == 0);
    CHECK(hl.clients[term].mapped == 1);
    CHECK(hl.clients[term].output_idx == 0);
    CHECK(hl.workspaces[0].active_clients == 1);
    CHECK(hl.workspaces[0].active == 1);
    /* map browser on workspace 2, output 1 */
    CHECK(wubu_hl_map_client(&hl, browser, 2, 1, 1920, 0, 1000, 700) == 0);
    CHECK(hl.workspaces[1].active_clients == 1);
    /* workspace 2 now active (last map switches active workspace) */
    CHECK(hl.workspaces[1].active == 1);
    CHECK(hl.workspaces[2].active == 0);

    /* client_at: pick the mapped client at a point */
    CHECK(wubu_hl_client_at(&hl, 100, 100) == term);       /* inside terminal rect */
    CHECK(wubu_hl_client_at(&hl, 2000, 500) == browser);    /* inside browser rect */
    CHECK(wubu_hl_client_at(&hl, 0, 700) == -1);           /* below both rects */
    /* map editor overlaps term -> topmost (editor) wins */
    CHECK(wubu_hl_map_client(&hl, editor, 1, 0, 10, 10, 800, 600) == 0);
    CHECK(wubu_hl_client_at(&hl, 50, 50) == editor);

    /* unmap editor */
    CHECK(wubu_hl_unmap_client(&hl, editor) == 0);
    CHECK(hl.clients[editor].mapped == 0);
    CHECK(hl.workspaces[0].active_clients == 1);
    CHECK(wubu_hl_client_at(&hl, 50, 50) == term);
    CHECK(wubu_hl_unmap_client(&hl, editor) == -1);  /* already unmapped */

    /* n_mapped: only term+browser mapped */
    CHECK(wubu_hl_n_mapped(&hl) == 2);

    /* handshake + dmabuf */
    CHECK(wubu_hl_handshake(&hl) == 0);
    CHECK(hl.handshake_done == 1);
    CHECK(hl.dmabuf_implicit == 1);
    CHECK(wubu_hl_set_dmabuf_implicit(&hl, 0) == 0);
    CHECK(hl.dmabuf_implicit == 0);

    /* bad map args */
    CHECK(wubu_hl_map_client(&hl, term, 5, 0, 0, 0, 100, 100) == -1);    /* bad ws */
    CHECK(wubu_hl_map_client(&hl, term, 1, 99, 0, 0, 100, 100) == -1);   /* bad output */
    CHECK(wubu_hl_map_client(&hl, 99, 1, 0, 0, 0, 100, 100) == -1);       /* bad cid */

    /* teardown: surfaces cleared */
    CHECK(wubu_hl_teardown(&hl) == 0);
    CHECK(wubu_hl_state(&hl) == WUBU_HL_STATE_DOWN);
    CHECK(wubu_hl_n_mapped(&hl) == 0);
    CHECK(hl.n_clients == 0);
    CHECK(hl.handshake_done == 0);
    /* double teardown fails */
    CHECK(wubu_hl_teardown(&hl) == -1);

    /* re-run a fresh compositor */
    CHECK(wubu_hl_init(&hl) == 0);
    CHECK(wubu_hl_run(&hl, "wayland-0") == 0);
    CHECK(wubu_hl_add_output(&hl, "HDMI-A-1", 3840, 2160, 144000) == 0);
    CHECK(wubu_hl_state(&hl) == WUBU_HL_STATE_RUNNING);

    /* bad args */
    CHECK(wubu_hl_add_client(NULL, "x", 1) == -1);
    int c0 = wubu_hl_add_client(&hl, "terminal", 2001); CHECK(c0 == 0);
    CHECK(wubu_hl_map_client(&hl, 0, 1, 0, 0, 0, 100, 100) == 0);  /* ok: 1 output */
    CHECK(wubu_hl_client_at(NULL, 0, 0) == -1);
    CHECK(wubu_hl_init(NULL) == -1);

    if (fails == 0) printf("ALL HYPRLAND TESTS PASSED\n");
    else printf("%d HYPRLAND TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
