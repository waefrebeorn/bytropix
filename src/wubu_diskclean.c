/* wubu_diskclean.c -- Win98/XP Disk Cleanup utility.
 * See wubu_diskclean.h. Real, deterministic disk-space analyzer. */

#include "wubu_diskclean.h"
#include <string.h>

static const char *g_names[WUBU_DC_NCATS] = {
    "Temporary files", "Cache", "Recycle Bin", "Logs",
    "Downloads", "Other"
};

const char *wubu_dc_cat_name(wubu_dc_cat_t cat) {
    if (cat < 0 || cat >= WUBU_DC_NCATS) return NULL;
    return g_names[cat];
}

int wubu_dc_init(wubu_diskclean_t *dc) {
    if (!dc) return -1;
    memset(dc, 0, sizeof(*dc));
    return 0;
}

int wubu_dc_add(wubu_diskclean_t *dc, const char *path, wubu_dc_cat_t cat,
                uint64_t size) {
    if (!dc || !path) return -1;
    if (cat < 0 || cat >= WUBU_DC_NCATS) return -1;
    if (dc->n_files >= WUBU_DC_MAX_FILES) return -1;
    wubu_dc_file_t *f = &dc->files[dc->n_files];
    memset(f, 0, sizeof(*f));
    strncpy(f->path, path, WUBU_DC_PATH_MAX - 1);
    f->cat = cat;
    f->size = size;
    dc->cat_bytes[cat] += size;
    dc->cat_count[cat]++;
    dc->n_files++;
    return 0;
}

int wubu_dc_safe(wubu_dc_cat_t cat) {
    switch (cat) {
    case WUBU_DC_TEMP:
    case WUBU_DC_CACHE:
    case WUBU_DC_RECYCLE_BIN:
    case WUBU_DC_LOG:
        return 1;
    default:
        return 0;   /* downloads need opt-in; other never auto-selected */
    }
}

uint64_t wubu_dc_cat_bytes(const wubu_diskclean_t *dc, wubu_dc_cat_t cat) {
    if (!dc || cat < 0 || cat >= WUBU_DC_NCATS) return 0;
    return dc->cat_bytes[cat];
}

uint64_t wubu_dc_reclaimable(const wubu_diskclean_t *dc, unsigned sel) {
    if (!dc) return 0;
    uint64_t total = 0;
    for (int i = 0; i < WUBU_DC_NCATS; i++)
        if (sel & (1u << i)) total += dc->cat_bytes[i];
    return total;
}

unsigned wubu_dc_default_selection(void) {
    return (1u << WUBU_DC_TEMP) | (1u << WUBU_DC_CACHE) |
           (1u << WUBU_DC_RECYCLE_BIN) | (1u << WUBU_DC_LOG);
}
