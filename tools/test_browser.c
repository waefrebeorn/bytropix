/* test_browser.c -- Win98/XP tabbed Web browser tests. */
#include "wubu_browser.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_browser_t b;

    /* init: one tab at home */
    CHECK(wubu_br_init(&b, "about:home") == 0);
    CHECK(wubu_br_n_tabs(&b) == 1);
    CHECK(b.active == 0);
    CHECK(strcmp(wubu_br_url(&b), "about:home") == 0);
    CHECK(wubu_br_can_back(&b) == 0);
    CHECK(wubu_br_can_forward(&b) == 0);
    CHECK(wubu_br_init(&b, NULL) == -1);
    CHECK(wubu_br_init(NULL, "x") == -1);

    /* navigate builds history */
    CHECK(wubu_br_navigate(&b, "http://a.com") == 0);
    CHECK(strcmp(wubu_br_url(&b), "http://a.com") == 0);
    CHECK(wubu_br_can_back(&b) == 1);
    CHECK(wubu_br_navigate(&b, "http://b.com") == 0);
    CHECK(wubu_br_can_back(&b) == 1);
    CHECK(wubu_br_navigate(&b, "http://c.com") == 0);
    CHECK(strcmp(wubu_br_url(&b), "http://c.com") == 0);

    /* back/forward */
    CHECK(wubu_br_back(&b) == 0);
    CHECK(strcmp(wubu_br_url(&b), "http://b.com") == 0);
    CHECK(wubu_br_can_forward(&b) == 1);
    CHECK(wubu_br_back(&b) == 0);
    CHECK(strcmp(wubu_br_url(&b), "http://a.com") == 0);
    CHECK(wubu_br_back(&b) == 0);
    CHECK(strcmp(wubu_br_url(&b), "about:home") == 0);
    CHECK(wubu_br_back(&b) == 1);   /* nowhere to go */
    /* forward again */
    CHECK(wubu_br_forward(&b) == 0);
    CHECK(strcmp(wubu_br_url(&b), "http://a.com") == 0);
    CHECK(wubu_br_forward(&b) == 0);
    CHECK(strcmp(wubu_br_url(&b), "http://b.com") == 0);
    CHECK(wubu_br_forward(&b) == 0);
    CHECK(strcmp(wubu_br_url(&b), "http://c.com") == 0);
    CHECK(wubu_br_forward(&b) == 1);   /* nowhere forward */

    /* a new navigation clears forward */
    CHECK(wubu_br_back(&b) == 0);   /* -> b.com, forward has c.com */
    CHECK(wubu_br_can_forward(&b) == 1);
    CHECK(wubu_br_navigate(&b, "http://d.com") == 0);
    CHECK(wubu_br_can_forward(&b) == 0);   /* forward cleared */
    CHECK(strcmp(wubu_br_url(&b), "http://d.com") == 0);

    /* tabs: independent history */
    int t1 = wubu_br_new_tab(&b, "http://tab2.com");
    CHECK(t1 == 1);
    CHECK(wubu_br_n_tabs(&b) == 2);
    CHECK(b.active == 1);
    CHECK(strcmp(wubu_br_url(&b), "http://tab2.com") == 0);
    CHECK(wubu_br_can_back(&b) == 0);   /* fresh tab, no history */
    /* switch back to tab 0: its history intact */
    CHECK(wubu_br_switch(&b, 0) == 0);
    CHECK(strcmp(wubu_br_url(&b), "http://d.com") == 0);
    CHECK(wubu_br_can_back(&b) == 1);
    /* tab 0 navigates independently */
    CHECK(wubu_br_navigate(&b, "http://e.com") == 0);
    CHECK(wubu_br_switch(&b, 1) == 0);
    CHECK(strcmp(wubu_br_url(&b), "http://tab2.com") == 0);   /* tab1 unaffected */

    /* bookmarks */
    CHECK(wubu_br_n_bookmarks(&b) == 0);
    CHECK(wubu_br_bookmark(&b, "http://a.com") == 0);
    CHECK(wubu_br_bookmark(&b, "http://b.com") == 0);
    CHECK(wubu_br_n_bookmarks(&b) == 2);
    CHECK(wubu_br_is_bookmarked(&b, "http://a.com") == 1);
    CHECK(wubu_br_is_bookmarked(&b, "http://zzz.com") == 0);
    CHECK(wubu_br_bookmark(&b, "http://a.com") == -1);   /* dup */
    CHECK(wubu_br_unbookmark(&b, "http://a.com") == 0);
    CHECK(wubu_br_is_bookmarked(&b, "http://a.com") == 0);
    CHECK(wubu_br_n_bookmarks(&b) == 1);
    CHECK(wubu_br_unbookmark(&b, "http://a.com") == -1);   /* already gone */
    CHECK(wubu_br_bookmark(&b, NULL) == -1);
    CHECK(wubu_br_unbookmark(&b, NULL) == -1);

    /* close tab: can't close the last one */
    CHECK(wubu_br_close_tab(&b, 0) == 0);   /* close tab 0 */
    CHECK(wubu_br_n_tabs(&b) == 1);
    CHECK(b.active == 0);
    CHECK(strcmp(wubu_br_url(&b), "http://tab2.com") == 0);   /* tab1 now active */
    CHECK(wubu_br_close_tab(&b, 0) == -1);   /* can't close last */
    CHECK(wubu_br_close_tab(&b, 99) == -1);
    CHECK(wubu_br_close_tab(NULL, 0) == -1);

    /* switch bad */
    CHECK(wubu_br_switch(&b, 99) == -1);
    CHECK(wubu_br_switch(NULL, 0) == -1);

    /* navigate/back/forward NULL */
    CHECK(wubu_br_navigate(NULL, "x") == -1);
    CHECK(wubu_br_navigate(&b, NULL) == -1);
    CHECK(wubu_br_back(NULL) == -1);
    CHECK(wubu_br_forward(NULL) == -1);
    CHECK(wubu_br_new_tab(NULL, "x") == -1);
    CHECK(wubu_br_new_tab(&b, NULL) == -1);

    /* query NULL */
    CHECK(wubu_br_url(NULL) == NULL);
    CHECK(wubu_br_can_back(NULL) == 0);
    CHECK(wubu_br_can_forward(NULL) == 0);
    CHECK(wubu_br_n_tabs(NULL) == -1);
    CHECK(wubu_br_is_bookmarked(NULL, "x") == 0);
    CHECK(wubu_br_n_bookmarks(NULL) == -1);

    if (fails == 0) printf("ALL BROWSER TESTS PASSED\n");
    else printf("%d BROWSER TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
