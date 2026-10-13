#ifndef WUBU_CHARMAP_H
#define WUBU_CHARMAP_H

/* wubu_charmap.h -- Win98 Character Map accessory.
 *
 * A navigable grid of Unicode codepoints within a selectable block range.
 * D-pad/arrow navigation moves a cell cursor across the grid (row stride =
 * column count); the focused cell's codepoint is shown with its UTF-8
 * encoding, and "select" copies the codepoint into a clipboard buffer that
 * accumulates multiple selections. Deterministic, no stubs. */

#include <stddef.h>
#include <stdint.h>

#define WUBU_CM_MAX_COLS 16
#define WUBU_CM_NAME_MAX 32
#define WUBU_CM_CLIP_MAX 64

typedef struct {
    char name[WUBU_CM_NAME_MAX];
    uint32_t start;      /* first codepoint in the block */
    uint32_t end;        /* last codepoint in the block */
} wubu_cm_block_t;

typedef struct {
    int block;           /* active block index */
    int cols;            /* grid columns */
    uint32_t cursor;     /* focused codepoint */
    char clipboard[WUBU_CM_CLIP_MAX];   /* accumulated UTF-8 selections */
    int clip_len;        /* bytes in clipboard */
    int n_selects;
} wubu_charmap_t;

/* Initialise with a grid column count (1..CM_MAX_COLS). Returns 0/-1. */
int wubu_cm_init(wubu_charmap_t *cm, int cols);

/* Select the active block by index into the built-in block table. Returns
 * 0/-1. Resets the cursor to the block start. */
int wubu_cm_set_block(wubu_charmap_t *cm, int block_idx);

/* Number of built-in blocks. */
int wubu_cm_n_blocks(void);

/* Block descriptor by index (NULL if out of range). */
const wubu_cm_block_t *wubu_cm_block(int idx);

/* Move the grid cursor by (drow,dcol) grid deltas; clamps within the active
 * block. Returns the new cursor codepoint, -1 on bad args. */
uint32_t wubu_cm_move(wubu_charmap_t *cm, int drow, int dcol);

/* Set the cursor to an absolute codepoint within the active block. */
int wubu_cm_set_cursor(wubu_charmap_t *cm, uint32_t cp);

/* Select the focused codepoint: append its UTF-8 to the clipboard. Returns
 * bytes appended, -1 on bad args / clipboard full. */
int wubu_cm_select(wubu_charmap_t *cm);

/* Clear the clipboard. */
int wubu_cm_clear_clipboard(wubu_charmap_t *cm);

/* Query. */
uint32_t wubu_cm_cursor(const wubu_charmap_t *cm);
const char *wubu_cm_clipboard(const wubu_charmap_t *cm);

/* Encode a codepoint to UTF-8 into buf (>=5 bytes). Returns bytes written,
 * -1 on bad args / invalid codepoint (surrogate or >0x10FFFF). */
int wubu_cm_utf8(uint32_t cp, char *buf);

#endif
