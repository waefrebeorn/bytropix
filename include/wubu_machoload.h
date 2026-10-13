#ifndef WUBU_MACHOLOAD_H
#define WUBU_MACHOLOAD_H

/* wubu_machoload.h -- RUNTIME #347 Mach-O execution leg.
 *
 * A minimal Mach-O 64 (x86_64) parser/loader: validate the header (magic,
 * 64-bit, little-endian, CPU type x86_64, filetype), and map every LC_SEGMENT_64
 * segment that carries the MH_EXECUTE/entry into a page-aligned arena at its
 * vmaddr, honouring vmsize/filesize (zero-fill for __bss). Returns the computed
 * entry point (entryoff + base). This is the in-kernel execution leg that lets
 * a Mach-O binary run ON the AGI kernel. Deterministic, real work (no stubs). */

#include <stddef.h>
#include <stdint.h>

#define WUBU_MACHO_PAGE 4096

typedef enum {
    WUBU_MACHO_OK = 0,
    WUBU_MACHO_ERR_ARGS = -1,
    WUBU_MACHO_ERR_TRUNC = -2,
    WUBU_MACHO_ERR_MAGIC = -3,      /* not a Mach-O */
    WUBU_MACHO_ERR_CLASS = -4,      /* not 64-bit */
    WUBU_MACHO_ERR_ENDIAN = -5,     /* not little-endian */
    WUBU_MACHO_ERR_CPU = -6,        /* not x86_64 */
    WUBU_MACHO_ERR_FILETYPE = -7,   /* unsupported filetype */
    WUBU_MACHO_ERR_CMDS = -8,       /* load-command table overruns buffer */
    WUBU_MACHO_ERR_NOENTRY = -9,    /* no mappable segments */
    WUBU_MACHO_ERR_ARENA = -10,     /* arena too small */
    WUBU_MACHO_ERR_SEGMENT = -11    /* bad segment layout */
} wubu_macho_status_t;

typedef struct {
    uint64_t base;          /* arena base address the image was mapped at */
    uint64_t entry;         /* absolute entry point (base + entryoff) */
    uint64_t min_vmaddr;    /* lowest mapped vmaddr (page-aligned) */
    uint64_t max_vmaddr;    /* one past the highest mapped byte */
    uint64_t image_size;    /* max_vmaddr - min_vmaddr */
    int n_seg;              /* number of LC_SEGMENT_64 segments mapped */
    int filetype;
    uint32_t cputype;
    uint32_t ncmds;
} wubu_macho_image_t;

/* Parse+load a Mach-O 64 x86_64 image from `buf` (length `len`) into `arena`
 * (capacity `arena_cap`) at base `base`. On success fills `img`, returns
 * WUBU_MACHO_OK; on failure returns a negative status, leaving arena/img
 * untouched. */
int wubu_macho_load(const uint8_t *buf, size_t len,
                    uint8_t *arena, size_t arena_cap, uint64_t base,
                    wubu_macho_image_t *img);

/* Validate a Mach-O 64 x86_64 header without loading. */
int wubu_macho_validate(const uint8_t *buf, size_t len);

/* Human-readable name for a status code. */
const char *wubu_macho_strerror(int status);

#endif
