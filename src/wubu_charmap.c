/* wubu_charmap.c -- Win98 Character Map accessory.
 * See wubu_charmap.h. Real, deterministic Unicode grid navigator. */

#include "wubu_charmap.h"
#include <string.h>

/* Built-in Unicode block table (a practical subset of common blocks). */
static const wubu_cm_block_t g_blocks[] = {
    { "Basic Latin",       0x0000, 0x007F },
    { "Latin-1 Supplement",0x0080, 0x00FF },
    { "Latin Extended-A",  0x0100, 0x017F },
    { "Greek",             0x0370, 0x03FF },
    { "Cyrillic",          0x0400, 0x04FF },
    { "Arrows",            0x2190, 0x21FF },
    { "Math Operators",    0x2200, 0x22FF },
    { "Box Drawing",       0x2500, 0x257F },
    { "Geometric Shapes",  0x25A0, 0x25FF },
    { "CJK Symbols",       0x3000, 0x303F },
    { "Hiragana",          0x3040, 0x309F },
    { "Katakana",          0x30A0, 0x30FF },
};
#define WUBU_CM_N_BLOCKS ((int)(sizeof(g_blocks)/sizeof(g_blocks[0])))

int wubu_cm_n_blocks(void) { return WUBU_CM_N_BLOCKS; }

const wubu_cm_block_t *wubu_cm_block(int idx) {
    if (idx < 0 || idx >= WUBU_CM_N_BLOCKS) return NULL;
    return &g_blocks[idx];
}

int wubu_cm_utf8(uint32_t cp, char *buf) {
    if (!buf) return -1;
    if (cp > 0x10FFFF) return -1;
    if (cp >= 0xD800 && cp <= 0xDFFF) return -1;   /* surrogate */
    if (cp < 0x80) {
        buf[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else {
        buf[0] = (char)(0xF0 | (cp >> 18));
        buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
}

int wubu_cm_init(wubu_charmap_t *cm, int cols) {
    if (!cm || cols < 1 || cols > WUBU_CM_MAX_COLS) return -1;
    memset(cm, 0, sizeof(*cm));
    cm->cols = cols;
    cm->block = 0;   /* Basic Latin */
    cm->cursor = g_blocks[0].start;
    return 0;
}

int wubu_cm_set_block(wubu_charmap_t *cm, int block_idx) {
    if (!cm || block_idx < 0 || block_idx >= WUBU_CM_N_BLOCKS) return -1;
    cm->block = block_idx;
    cm->cursor = g_blocks[block_idx].start;
    return 0;
}

uint32_t wubu_cm_move(wubu_charmap_t *cm, int drow, int dcol) {
    if (!cm) return (uint32_t)-1;
    const wubu_cm_block_t *b = &g_blocks[cm->block];
    uint32_t cur = cm->cursor;
    /* grid delta: down/up move by a row (cols), left/right by one */
    long target = (long)cur + (long)drow * cm->cols + dcol;
    if (target < (long)b->start) target = (long)b->start;
    if (target > (long)b->end) target = (long)b->end;
    cm->cursor = (uint32_t)target;
    return cm->cursor;
}

int wubu_cm_set_cursor(wubu_charmap_t *cm, uint32_t cp) {
    if (!cm) return -1;
    const wubu_cm_block_t *b = &g_blocks[cm->block];
    if (cp < b->start || cp > b->end) return -1;
    cm->cursor = cp;
    return 0;
}

int wubu_cm_select(wubu_charmap_t *cm) {
    if (!cm) return -1;
    char buf[5];
    int n = wubu_cm_utf8(cm->cursor, buf);
    if (n < 0) return -1;
    if (cm->clip_len + n >= WUBU_CM_CLIP_MAX) return -1;   /* full */
    memcpy(cm->clipboard + cm->clip_len, buf, n);
    cm->clip_len += n;
    cm->clipboard[cm->clip_len] = '\0';
    cm->n_selects++;
    return n;
}

int wubu_cm_clear_clipboard(wubu_charmap_t *cm) {
    if (!cm) return -1;
    cm->clip_len = 0;
    cm->clipboard[0] = '\0';
    return 0;
}

uint32_t wubu_cm_cursor(const wubu_charmap_t *cm) {
    if (!cm) return (uint32_t)-1;
    return cm->cursor;
}

const char *wubu_cm_clipboard(const wubu_charmap_t *cm) {
    if (!cm) return NULL;
    return cm->clipboard;
}
