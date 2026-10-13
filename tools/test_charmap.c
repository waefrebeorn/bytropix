/* test_charmap.c -- Win98 Character Map accessory tests. */
#include "wubu_charmap.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_charmap_t cm;

    /* init: 16-col grid, Basic Latin block */
    CHECK(wubu_cm_init(&cm, 16) == 0);
    CHECK(cm.cols == 16);
    CHECK(cm.block == 0);
    CHECK(wubu_cm_cursor(&cm) == 0x0000);   /* Basic Latin start */
    CHECK(strcmp(wubu_cm_clipboard(&cm), "") == 0);
    CHECK(wubu_cm_init(&cm, 0) == -1);
    CHECK(wubu_cm_init(&cm, WUBU_CM_MAX_COLS + 1) == -1);
    CHECK(wubu_cm_init(NULL, 16) == -1);

    /* built-in blocks */
    CHECK(wubu_cm_n_blocks() > 0);
    const wubu_cm_block_t *b = wubu_cm_block(0);
    CHECK(b != NULL);
    CHECK(strcmp(b->name, "Basic Latin") == 0);
    CHECK(b->start == 0x0000 && b->end == 0x007F);
    CHECK(wubu_cm_block(999) == NULL);
    CHECK(wubu_cm_block(-1) == NULL);

    /* UTF-8 encoding correctness */
    char buf[5];
    CHECK(wubu_cm_utf8('A', buf) == 1 && buf[0] == 'A');
    CHECK(wubu_cm_utf8(0x00E9, buf) == 2);   /* e-acute, 2 bytes */
    CHECK((unsigned char)buf[0] == 0xC3 && (unsigned char)buf[1] == 0xA9);
    CHECK(wubu_cm_utf8(0x20AC, buf) == 3);   /* euro, 3 bytes */
    CHECK((unsigned char)buf[0] == 0xE2 && (unsigned char)buf[1] == 0x82 && (unsigned char)buf[2] == 0xAC);
    CHECK(wubu_cm_utf8(0x1F600, buf) == 4);  /* emoji, 4 bytes */
    CHECK((unsigned char)buf[0] == 0xF0);
    CHECK(wubu_cm_utf8(0xD800, buf) == -1);  /* surrogate rejected */
    CHECK(wubu_cm_utf8(0x110000, buf) == -1); /* out of range */
    CHECK(wubu_cm_utf8('A', NULL) == -1);

    /* grid navigation: 16 cols, Basic Latin 0x00-0x7F */
    CHECK(wubu_cm_move(&cm, 0, 5) == 0x05);    /* right 5 */
    CHECK(wubu_cm_move(&cm, 1, 0) == 0x15);    /* down a row (+16) */
    CHECK(wubu_cm_move(&cm, 0, 1) == 0x16);
    CHECK(wubu_cm_move(&cm, -1, 0) == 0x06);   /* up a row (-16) */
    CHECK(wubu_cm_move(&cm, 0, -100) == 0x00); /* clamp to block start */
    CHECK(wubu_cm_move(&cm, 0, 1000) == 0x7F); /* clamp to block end */
    CHECK(wubu_cm_move(NULL, 0, 1) == (uint32_t)-1);

    /* set_cursor within block */
    CHECK(wubu_cm_set_cursor(&cm, 0x41) == 0);   /* 'A' */
    CHECK(wubu_cm_cursor(&cm) == 0x41);
    CHECK(wubu_cm_set_cursor(&cm, 0x100) == -1); /* out of Basic Latin */
    CHECK(wubu_cm_set_cursor(NULL, 0x41) == -1);

    /* select: copies UTF-8 to clipboard */
    CHECK(wubu_cm_set_cursor(&cm, 'A') == 0);
    CHECK(wubu_cm_select(&cm) == 1);
    CHECK(strcmp(wubu_cm_clipboard(&cm), "A") == 0);
    CHECK(wubu_cm_set_cursor(&cm, 'B') == 0);
    CHECK(wubu_cm_select(&cm) == 1);
    CHECK(strcmp(wubu_cm_clipboard(&cm), "AB") == 0);
    CHECK(cm.n_selects == 2);

    /* switch to Greek block, select a Greek letter */
    int greek = -1;
    for (int i = 0; i < wubu_cm_n_blocks(); i++) {
        const wubu_cm_block_t *bb = wubu_cm_block(i);
        if (bb && strcmp(bb->name, "Greek") == 0) { greek = i; break; }
    }
    CHECK(greek >= 0);
    CHECK(wubu_cm_set_block(&cm, greek) == 0);
    CHECK(wubu_cm_cursor(&cm) == 0x0370);   /* Greek start */
    CHECK(wubu_cm_set_cursor(&cm, 0x03B1) == 0);   /* alpha */
    CHECK(wubu_cm_select(&cm) == 2);        /* 2-byte UTF-8 */
    /* clipboard now "AB" + alpha(2 bytes) = 4 bytes */
    CHECK(cm.clip_len == 4);
    CHECK(wubu_cm_set_block(&cm, 999) == -1);
    CHECK(wubu_cm_set_block(&cm, -1) == -1);
    CHECK(wubu_cm_set_block(NULL, 0) == -1);

    /* clear clipboard */
    CHECK(wubu_cm_clear_clipboard(&cm) == 0);
    CHECK(strcmp(wubu_cm_clipboard(&cm), "") == 0);
    CHECK(cm.clip_len == 0);
    CHECK(wubu_cm_clear_clipboard(NULL) == -1);

    /* clipboard full guard: fill then select fails */
    for (int i = 0; i < WUBU_CM_CLIP_MAX; i++) {
        if (wubu_cm_set_cursor(&cm, 0x03B1) != 0) break;
        if (wubu_cm_select(&cm) < 0) break;   /* stop when full */
    }
    CHECK(cm.clip_len < WUBU_CM_CLIP_MAX);   /* never overflowed */
    CHECK(wubu_cm_select(&cm) >= -1);        /* no crash near full */

    /* bad args */
    CHECK(wubu_cm_select(NULL) == -1);
    CHECK(wubu_cm_cursor(NULL) == (uint32_t)-1);
    CHECK(wubu_cm_clipboard(NULL) == NULL);

    if (fails == 0) printf("ALL CHARMAP TESTS PASSED\n");
    else printf("%d CHARMAP TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
