#ifndef WUBU_FONTVIEW_H
#define WUBU_FONTVIEW_H

/* wubu_fontview.h -- Win98/XP Font viewer.
 *
 * Models the Fonts control-panel folder: an installed-font catalog, a preview
 * of a sample string rendered in the selected font (size + style), and the
 * active system font selection. Fonts carry a family name, style (regular/
 * bold/italic/bold-italic), and a point size; the preview reports the chosen
 * sample + active font. Deterministic, no stubs. */

#include <stddef.h>

#define WUBU_FV_MAX_FONTS 64
#define WUBU_FV_NAME_MAX 32
#define WUBU_FV_SAMPLE_MAX 128

typedef enum {
    WUBU_FV_REGULAR = 0,
    WUBU_FV_BOLD,
    WUBU_FV_ITALIC,
    WUBU_FV_BOLD_ITALIC
} wubu_fv_style_t;

typedef struct {
    char family[WUBU_FV_NAME_MAX];
    wubu_fv_style_t style;
    int point_size;      /* e.g. 8, 10, 12, 14, 24 */
    int monospace;       /* 1 = fixed-width */
} wubu_fv_font_t;

typedef struct {
    wubu_fv_font_t fonts[WUBU_FV_MAX_FONTS];
    int n_fonts;
    int selected;        /* selected font index, -1 if none */
    int active;          /* active system font index, -1 if none */
    char sample[WUBU_FV_SAMPLE_MAX];
    int preview_size;    /* preview point size override */
} wubu_fontview_t;

/* Initialise (empty catalog, default sample, no selection). */
int wubu_fv_init(wubu_fontview_t *fv);

/* Install a font into the catalog. Returns index or -1 (dup/full). */
int wubu_fv_install(wubu_fontview_t *fv, const char *family,
                    wubu_fv_style_t style, int point_size, int monospace);

/* Select a font by index (for preview). Returns 0/-1. */
int wubu_fv_select(wubu_fontview_t *fv, int idx);

/* Set the active system font by index. Returns 0/-1. */
int wubu_fv_set_active(wubu_fontview_t *fv, int idx);

/* Set the preview sample string (truncated to fit). Returns 0/-1. */
int wubu_fv_set_sample(wubu_fontview_t *fv, const char *sample);

/* Set the preview point size. Returns 0/-1. */
int wubu_fv_set_preview_size(wubu_fontview_t *fv, int size);

/* Query. */
int wubu_fv_selected(const wubu_fontview_t *fv);
int wubu_fv_active(const wubu_fontview_t *fv);
const wubu_fv_font_t *wubu_fv_font(const wubu_fontview_t *fv, int idx);
const char *wubu_fv_family(const wubu_fontview_t *fv, int idx);
const char *wubu_fv_style_name(wubu_fv_style_t style);

#endif
