/* wubu_fontview.c -- Win98/XP Font viewer.
 * See wubu_fontview.h. Real, deterministic font catalog + preview. */

#include "wubu_fontview.h"
#include <string.h>

static const char *g_style_names[] = {
    "Regular", "Bold", "Italic", "Bold Italic"
};

const char *wubu_fv_style_name(wubu_fv_style_t style) {
    if (style < 0 || style > WUBU_FV_BOLD_ITALIC) return NULL;
    return g_style_names[style];
}

int wubu_fv_init(wubu_fontview_t *fv) {
    if (!fv) return -1;
    memset(fv, 0, sizeof(*fv));
    fv->selected = -1;
    fv->active = -1;
    strncpy(fv->sample, "The quick brown fox", WUBU_FV_SAMPLE_MAX - 1);
    fv->preview_size = 24;
    return 0;
}

int wubu_fv_install(wubu_fontview_t *fv, const char *family,
                    wubu_fv_style_t style, int point_size, int monospace) {
    if (!fv || !family || point_size <= 0) return -1;
    if (style < 0 || style > WUBU_FV_BOLD_ITALIC) return -1;
    if (fv->n_fonts >= WUBU_FV_MAX_FONTS) return -1;
    wubu_fv_font_t *f = &fv->fonts[fv->n_fonts];
    memset(f, 0, sizeof(*f));
    strncpy(f->family, family, WUBU_FV_NAME_MAX - 1);
    f->style = style;
    f->point_size = point_size;
    f->monospace = monospace ? 1 : 0;
    int idx = fv->n_fonts++;
    if (fv->selected < 0) fv->selected = 0;   /* select first installed */
    return idx;
}

int wubu_fv_select(wubu_fontview_t *fv, int idx) {
    if (!fv || idx < 0 || idx >= fv->n_fonts) return -1;
    fv->selected = idx;
    return 0;
}

int wubu_fv_set_active(wubu_fontview_t *fv, int idx) {
    if (!fv || idx < 0 || idx >= fv->n_fonts) return -1;
    fv->active = idx;
    return 0;
}

int wubu_fv_set_sample(wubu_fontview_t *fv, const char *sample) {
    if (!fv || !sample) return -1;
    strncpy(fv->sample, sample, WUBU_FV_SAMPLE_MAX - 1);
    fv->sample[WUBU_FV_SAMPLE_MAX - 1] = '\0';
    return 0;
}

int wubu_fv_set_preview_size(wubu_fontview_t *fv, int size) {
    if (!fv || size <= 0) return -1;
    fv->preview_size = size;
    return 0;
}

int wubu_fv_selected(const wubu_fontview_t *fv) {
    if (!fv) return -1;
    return fv->selected;
}

int wubu_fv_active(const wubu_fontview_t *fv) {
    if (!fv) return -1;
    return fv->active;
}

const wubu_fv_font_t *wubu_fv_font(const wubu_fontview_t *fv, int idx) {
    if (!fv || idx < 0 || idx >= fv->n_fonts) return NULL;
    return &fv->fonts[idx];
}

const char *wubu_fv_family(const wubu_fontview_t *fv, int idx) {
    const wubu_fv_font_t *f = wubu_fv_font(fv, idx);
    if (!f) return NULL;
    return f->family;
}
