/* test_ctxmenu.c -- UX #243 right-click context menus everywhere.
 * Proves: add items + separator, open highlights first enabled, navigation
 * skips disabled + separators, wrap-around, select returns id and closes,
 * disabled item not selectable, fail-closed on NULL/closed. */
#include "wubu_ctxmenu.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_ctxmenu_t m;
    CHECK(wubu_ctx_init(&m) == 0, "init ok");

    /* build a menu: Open, separator, Cut(disabled), Copy, Paste */
    wubu_ctx_add(&m, "Open", 10);
    wubu_ctx_add_sep(&m);
    wubu_ctx_add(&m, "Cut", 20);
    wubu_ctx_add(&m, "Copy", 30);
    wubu_ctx_add(&m, "Paste", 40);
    CHECK(m.n == 5, "5 entries (4 items + separator)");
    CHECK(wubu_ctx_set_enabled(&m, 20, 0) == 0, "disable Cut");

    /* open highlights first enabled (Open) */
    CHECK(wubu_ctx_open(&m) == 0, "open ok");
    CHECK(m.cursor == 0, "cursor on Open (first enabled)");

    /* next skips separator + disabled Cut -> Copy */
    CHECK(wubu_ctx_next(&m) == 3, "next skips sep+disabled -> Copy (idx 3)");
    CHECK(wubu_ctx_next(&m) == 4, "next -> Paste");
    /* wrap around */
    CHECK(wubu_ctx_next(&m) == 0, "next wraps to Open");

    /* prev wraps to last enabled */
    CHECK(wubu_ctx_prev(&m) == 4, "prev wraps to Paste");

    /* select Copy */
    m.cursor = 3;
    CHECK(wubu_ctx_select(&m) == 30, "select Copy -> id 30");
    CHECK(m.open == 0, "menu closed after select");

    /* disabled item not selectable: force cursor to Cut, select fails */
    wubu_ctx_open(&m);
    m.cursor = 2;   /* Cut (disabled) */
    CHECK(wubu_ctx_select(&m) == -1, "disabled item not selectable");

    /* separator not selectable */
    m.cursor = 1;   /* separator */
    CHECK(wubu_ctx_select(&m) == -1, "separator not selectable");

    /* close */
    wubu_ctx_open(&m);
    CHECK(wubu_ctx_close(&m) == 0, "close ok");
    CHECK(m.open == 0, "closed");

    /* fail-closed */
    CHECK(wubu_ctx_next(NULL) == -1, "NULL next -> -1");
    CHECK(wubu_ctx_select(NULL) == -1, "NULL select -> -1");
    CHECK(wubu_ctx_add(NULL, "x", 1) == -1, "NULL add -> -1");
    CHECK(wubu_ctx_set_enabled(&m, 999, 0) == -1, "set_enabled unknown -> -1");

    printf("\n=== test_ctxmenu: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
