#ifndef WUBU_WORDPROC_H
#define WUBU_WORDPROC_H

/* wubu_wordproc.h -- Win98/XP Word processor (rich text).
 *
 * A rich-text document buffer: a sequence of styled character runs. The cursor
 * supports insert/delete at a position; text carries per-character style flags
 * (bold, italic, underline). Style toggles apply to a selection range (or the
 * next inserted char when nothing is selected). Tracks word + character counts
 * and the dirty (modified) flag. Deterministic, no stubs. */

#include <stddef.h>
#include <stdint.h>

#define WUBU_WP_MAX_TEXT 4096
#define WUBU_WP_BOLD   (1u << 0)
#define WUBU_WP_ITALIC (1u << 1)
#define WUBU_WP_UNDER  (1u << 2)

typedef struct {
    char ch;
    uint8_t style;   /* WUBU_WP_* bitmask */
} wubu_wp_char_t;

typedef struct {
    wubu_wp_char_t buf[WUBU_WP_MAX_TEXT];
    int len;
    int cursor;          /* insertion point (0..len) */
    int sel_anchor;      /* selection anchor (-1 = no selection) */
    int sel_end;         /* selection end */
    uint8_t pending_style; /* style for the next inserted char */
    int dirty;           /* modified since last save */
} wubu_wordproc_t;

/* Initialise (empty, cursor 0, no selection). */
int wubu_wp_init(wubu_wordproc_t *wp);

/* Insert a string at the cursor with the pending style. Returns chars
 * inserted, -1 on bad args / overflow. Advances the cursor. */
int wubu_wp_insert(wubu_wordproc_t *wp, const char *text);

/* Delete `n` chars starting at the cursor (or the selection if one is set).
 * Returns chars deleted, -1 on bad args. */
int wubu_wp_delete(wubu_wordproc_t *wp, int n);

/* Move the cursor to an absolute position (clamped 0..len). */
int wubu_wp_move_cursor(wubu_wordproc_t *wp, int pos);

/* Set a selection [start,end] (clamped, ordered). -1 clears it. */
int wubu_wp_select(wubu_wordproc_t *wp, int start, int end);

/* Toggle a style bit on the selection (or set as pending if no selection).
 * Returns 0/-1. */
int wubu_wp_toggle_style(wubu_wordproc_t *wp, uint8_t style_bit);

/* Clear the dirty flag (after save). */
int wubu_wp_mark_saved(wubu_wordproc_t *wp);

/* Query. */
int wubu_wp_len(const wubu_wordproc_t *wp);
int wubu_wp_cursor(const wubu_wordproc_t *wp);
int wubu_wp_words(const wubu_wordproc_t *wp);
int wubu_wp_dirty(const wubu_wordproc_t *wp);
uint8_t wubu_wp_style_at(const wubu_wordproc_t *wp, int pos);
/* Copy the plain text (no style) into buf. Returns chars written, -1. */
int wubu_wp_text(const wubu_wordproc_t *wp, char *buf, int cap);

#endif
