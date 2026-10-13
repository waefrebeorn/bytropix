/* test_clipboard.c -- UX #245 clipboard copy/cut/paste across apps.
 * Proves: copy then paste (repeatable), cut then paste (consumed), peek
 * doesn't consume, cross-app ownership, empty state, fail-closed on NULL. */
#include "wubu_clipboard.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    char buf[WUBU_CLIP_LEN];
    wubu_clipboard_t c;
    CHECK(wubu_clip_init(&c) == 0, "init ok");
    CHECK(wubu_clip_empty(&c) == 1, "starts empty");
    CHECK(wubu_clip_paste(&c, buf, sizeof buf) == -1, "paste empty -> -1");

    /* app 1 copies text */
    CHECK(wubu_clip_set(&c, "hello", WUBU_CLIP_COPY, 1) == 0, "app1 copy");
    CHECK(wubu_clip_empty(&c) == 0, "not empty");
    /* peek doesn't consume */
    CHECK(wubu_clip_peek(&c, buf, sizeof buf) == WUBU_CLIP_COPY, "peek returns COPY");
    CHECK(strcmp(buf, "hello") == 0, "peek content");
    /* app 2 pastes (cross-app) */
    CHECK(wubu_clip_paste(&c, buf, sizeof buf) == WUBU_CLIP_COPY, "app2 paste COPY");
    CHECK(strcmp(buf, "hello") == 0, "paste content");
    /* copy is repeatable */
    CHECK(wubu_clip_paste(&c, buf, sizeof buf) == WUBU_CLIP_COPY, "copy repeatable");
    CHECK(strcmp(buf, "hello") == 0, "copy content again");

    /* app 1 cuts text */
    CHECK(wubu_clip_set(&c, "cutme", WUBU_CLIP_CUT, 1) == 0, "app1 cut");
    CHECK(wubu_clip_paste(&c, buf, sizeof buf) == WUBU_CLIP_CUT, "paste CUT");
    CHECK(strcmp(buf, "cutme") == 0, "cut content");
    /* cut consumed: next paste is empty */
    CHECK(wubu_clip_empty(&c) == 1, "cut consumed -> empty");
    CHECK(wubu_clip_paste(&c, buf, sizeof buf) == -1, "paste after cut -> -1");

    /* history: after cut consumed, older copy still in history? our model
     * keeps slots; peek falls back to head. After cut, head invalid -> empty.
     * A new copy works. */
    CHECK(wubu_clip_set(&c, "again", WUBU_CLIP_COPY, 2) == 0, "app2 copy after");
    CHECK(wubu_clip_paste(&c, buf, sizeof buf) == WUBU_CLIP_COPY, "paste new copy");
    CHECK(strcmp(buf, "again") == 0, "new copy content");

    /* fail-closed */
    CHECK(wubu_clip_set(&c, NULL, WUBU_CLIP_COPY, 1) == -1, "NULL text -> -1");
    CHECK(wubu_clip_set(NULL, "x", WUBU_CLIP_COPY, 1) == -1, "NULL clip -> -1");
    CHECK(wubu_clip_peek(NULL, buf, sizeof buf) == -1, "NULL peek -> -1");

    printf("\n=== test_clipboard: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
