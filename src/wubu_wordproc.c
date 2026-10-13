/* wubu_wordproc.c -- Win98/XP Word processor (rich text).
 * See wubu_wordproc.h. Real, deterministic rich-text document buffer. */

#include "wubu_wordproc.h"
#include <string.h>

int wubu_wp_init(wubu_wordproc_t *wp) {
    if (!wp) return -1;
    memset(wp, 0, sizeof(*wp));
    wp->len = 0;
    wp->cursor = 0;
    wp->sel_anchor = -1;
    wp->sel_end = -1;
    wp->pending_style = 0;
    wp->dirty = 0;
    return 0;
}

int wubu_wp_insert(wubu_wordproc_t *wp, const char *text) {
    if (!wp || !text) return -1;
    int n = (int)strlen(text);
    if (wp->len + n > WUBU_WP_MAX_TEXT) return -1;   /* overflow */
    /* if a selection is set, delete it first (replace) */
    if (wp->sel_anchor >= 0) {
        int s = wp->sel_anchor < wp->sel_end ? wp->sel_anchor : wp->sel_end;
        int e = wp->sel_anchor < wp->sel_end ? wp->sel_end : wp->sel_anchor;
        int del = e - s;
        memmove(&wp->buf[s], &wp->buf[e], (wp->len - e) * sizeof(wubu_wp_char_t));
        wp->len -= del;
        wp->cursor = s;
        wp->sel_anchor = wp->sel_end = -1;
    }
    int c = wp->cursor;
    /* shift tail right */
    memmove(&wp->buf[c + n], &wp->buf[c], (wp->len - c) * sizeof(wubu_wp_char_t));
    for (int i = 0; i < n; i++) {
        wp->buf[c + i].ch = text[i];
        wp->buf[c + i].style = wp->pending_style;
    }
    wp->len += n;
    wp->cursor += n;
    wp->dirty = 1;
    return n;
}

int wubu_wp_delete(wubu_wordproc_t *wp, int n) {
    if (!wp || n < 0) return -1;
    if (wp->sel_anchor >= 0) {
        int s = wp->sel_anchor < wp->sel_end ? wp->sel_anchor : wp->sel_end;
        int e = wp->sel_anchor < wp->sel_end ? wp->sel_end : wp->sel_anchor;
        int del = e - s;
        memmove(&wp->buf[s], &wp->buf[e], (wp->len - e) * sizeof(wubu_wp_char_t));
        wp->len -= del;
        wp->cursor = s;
        wp->sel_anchor = wp->sel_end = -1;
        wp->dirty = 1;
        return del;
    }
    /* delete n chars forward from the cursor */
    if (n > wp->len - wp->cursor) n = wp->len - wp->cursor;
    if (n == 0) return 0;
    memmove(&wp->buf[wp->cursor], &wp->buf[wp->cursor + n],
            (wp->len - wp->cursor - n) * sizeof(wubu_wp_char_t));
    wp->len -= n;
    wp->dirty = 1;
    return n;
}

int wubu_wp_move_cursor(wubu_wordproc_t *wp, int pos) {
    if (!wp) return -1;
    if (pos < 0) pos = 0;
    if (pos > wp->len) pos = wp->len;
    wp->cursor = pos;
    return 0;
}

int wubu_wp_select(wubu_wordproc_t *wp, int start, int end) {
    if (!wp) return -1;
    if (start < 0 || end < 0) { wp->sel_anchor = wp->sel_end = -1; return 0; }
    if (start > wp->len) start = wp->len;
    if (end > wp->len) end = wp->len;
    wp->sel_anchor = start;
    wp->sel_end = end;
    wp->cursor = end;
    return 0;
}

int wubu_wp_toggle_style(wubu_wordproc_t *wp, uint8_t style_bit) {
    if (!wp) return -1;
    if (style_bit & ~(WUBU_WP_BOLD | WUBU_WP_ITALIC | WUBU_WP_UNDER)) return -1;
    if (wp->sel_anchor < 0) {
        /* no selection: set as pending style for the next insert */
        wp->pending_style ^= style_bit;
        return 0;
    }
    int s = wp->sel_anchor < wp->sel_end ? wp->sel_anchor : wp->sel_end;
    int e = wp->sel_anchor < wp->sel_end ? wp->sel_end : wp->sel_anchor;
    for (int i = s; i < e; i++)
        wp->buf[i].style ^= style_bit;
    wp->dirty = 1;
    return 0;
}

int wubu_wp_mark_saved(wubu_wordproc_t *wp) {
    if (!wp) return -1;
    wp->dirty = 0;
    return 0;
}

int wubu_wp_len(const wubu_wordproc_t *wp) {
    return wp ? wp->len : -1;
}

int wubu_wp_cursor(const wubu_wordproc_t *wp) {
    return wp ? wp->cursor : -1;
}

int wubu_wp_words(const wubu_wordproc_t *wp) {
    if (!wp) return -1;
    int words = 0, in_word = 0;
    for (int i = 0; i < wp->len; i++) {
        char c = wp->buf[i].ch;
        if (c == ' ' || c == '\t' || c == '\n') {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            words++;
        }
    }
    return words;
}

int wubu_wp_dirty(const wubu_wordproc_t *wp) {
    return wp ? wp->dirty : -1;
}

uint8_t wubu_wp_style_at(const wubu_wordproc_t *wp, int pos) {
    if (!wp || pos < 0 || pos >= wp->len) return 0;
    return wp->buf[pos].style;
}

int wubu_wp_text(const wubu_wordproc_t *wp, char *buf, int cap) {
    if (!wp || !buf || cap <= 0) return -1;
    int n = wp->len < cap - 1 ? wp->len : cap - 1;
    for (int i = 0; i < n; i++) buf[i] = wp->buf[i].ch;
    buf[n] = '\0';
    return n;
}
