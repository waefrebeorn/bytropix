/* test_dnd.c -- UX #244 drag-and-drop between windows.
 * Proves: full drag lifecycle (begin->hover->drop accept -> DROPPED), reject
 * -> CANCELLED, cancel via Escape, drop-without-target rejected, payload
 * carried, fail-closed on bad state/NULL. */
#include "wubu_dnd.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_dnd_t d;
    CHECK(wubu_dnd_init(&d) == 0, "init ok");
    CHECK(wubu_dnd_get_state(&d) == WUBU_DND_IDLE, "starts idle");

    /* cannot hover/drop when idle */
    CHECK(wubu_dnd_hover(&d, 2) == -1, "hover when idle -> -1");
    CHECK(wubu_dnd_drop(&d, 1) == -1, "drop when idle -> -1");

    /* begin drag from window 1 with payload */
    CHECK(wubu_dnd_begin(&d, 1, "file.txt") == 0, "begin drag");
    CHECK(wubu_dnd_get_state(&d) == WUBU_DND_DRAGGING, "dragging");
    CHECK(strcmp(d.payload, "file.txt") == 0, "payload carried");

    /* hover over window 2 */
    CHECK(wubu_dnd_hover(&d, 2) == 0, "hover window 2");
    CHECK(d.dst_window == 2, "dst set");

    /* drop accepted -> DROPPED */
    CHECK(wubu_dnd_drop(&d, 1) == WUBU_DND_DROPPED, "accept -> DROPPED");
    CHECK(d.accepts == 1, "accepts flag");

    /* new drag, drop rejected -> CANCELLED */
    wubu_dnd_begin(&d, 1, "file.txt");
    wubu_dnd_hover(&d, 3);
    CHECK(wubu_dnd_drop(&d, 0) == WUBU_DND_CANCELLED, "reject -> CANCELLED");
    CHECK(d.accepts == 0, "not accepted");

    /* cancel via Escape */
    wubu_dnd_begin(&d, 1, "x");
    CHECK(wubu_dnd_cancel(&d) == 0, "cancel ok");
    CHECK(wubu_dnd_get_state(&d) == WUBU_DND_CANCELLED, "cancelled");

    /* drop with no target (dst=-1) -> rejected even if accept=1 */
    wubu_dnd_begin(&d, 1, "x");
    CHECK(wubu_dnd_drop(&d, 1) == WUBU_DND_CANCELLED, "drop no-target -> CANCELLED");

    /* cannot begin twice / drop twice */
    wubu_dnd_begin(&d, 1, "a");
    wubu_dnd_hover(&d, 5);
    CHECK(wubu_dnd_drop(&d, 1) == WUBU_DND_DROPPED, "first drop");
    CHECK(wubu_dnd_drop(&d, 1) == -1, "second drop (not dragging) -> -1");
    CHECK(wubu_dnd_cancel(&d) == -1, "cancel when not dragging -> -1");

    /* fail-closed */
    CHECK(wubu_dnd_begin(&d, 1, NULL) == -1, "NULL payload -> -1");
    CHECK(wubu_dnd_begin(NULL, 1, "x") == -1, "NULL dnd -> -1");
    CHECK(wubu_dnd_get_state(NULL) == WUBU_DND_IDLE, "NULL state -> IDLE");

    printf("\n=== test_dnd: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
