#ifndef WUBU_DISKCLEAN_H
#define WUBU_DISKCLEAN_H

/* wubu_diskclean.h -- Win98/XP Disk Cleanup utility.
 *
 * Scans an inventory of files, each tagged with a category (temp / cache /
 * recycle-bin / log / download / other), and reports reclaimable space by
 * category. Only "safe" categories (temp, cache, recycle-bin, log) are offered
 * for deletion; downloads require explicit opt-in. Produces a per-category
 * byte total + a total-reclaimable figure, and a selection mask drives which
 * categories a cleanup run would remove. Deterministic, no stubs. */

#include <stddef.h>
#include <stdint.h>

#define WUBU_DC_MAX_FILES 512
#define WUBU_DC_PATH_MAX 96
#define WUBU_DC_NCATS 6

typedef enum {
    WUBU_DC_TEMP = 0,        /* safe to delete */
    WUBU_DC_CACHE,           /* safe to delete */
    WUBU_DC_RECYCLE_BIN,     /* safe to delete */
    WUBU_DC_LOG,             /* safe to delete */
    WUBU_DC_DOWNLOAD,        /* requires opt-in */
    WUBU_DC_OTHER            /* never auto-selected */
} wubu_dc_cat_t;

typedef struct {
    char path[WUBU_DC_PATH_MAX];
    wubu_dc_cat_t cat;
    uint64_t size;
} wubu_dc_file_t;

typedef struct {
    wubu_dc_file_t files[WUBU_DC_MAX_FILES];
    int n_files;
    uint64_t cat_bytes[WUBU_DC_NCATS];     /* total bytes per category */
    int cat_count[WUBU_DC_NCATS];          /* file count per category */
} wubu_diskclean_t;

/* Initialise (empty). */
int wubu_dc_init(wubu_diskclean_t *dc);

/* Add a file to the inventory. Returns 0/-1 (full/bad cat). */
int wubu_dc_add(wubu_diskclean_t *dc, const char *path, wubu_dc_cat_t cat,
                uint64_t size);

/* Is a category safe to auto-select for deletion? 1/0. */
int wubu_dc_safe(wubu_dc_cat_t cat);

/* Bytes reclaimable if `sel` categories are cleaned. `sel` is a bitmask of
 * (1<<cat). Returns the total bytes. */
uint64_t wubu_dc_reclaimable(const wubu_diskclean_t *dc, unsigned sel);

/* Bytes for a single category. */
uint64_t wubu_dc_cat_bytes(const wubu_diskclean_t *dc, wubu_dc_cat_t cat);

/* Build the default "safe" selection mask (temp+cache+recycle+log). */
unsigned wubu_dc_default_selection(void);

/* Category name (NULL if invalid). */
const char *wubu_dc_cat_name(wubu_dc_cat_t cat);

#endif
