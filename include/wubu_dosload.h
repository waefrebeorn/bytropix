#ifndef WUBU_DOSLOAD_H
#define WUBU_DOSLOAD_H

/* wubu_dosload.h -- RUNTIME #345 DOS execution leg.
 *
 * A minimal MZ (DOS .EXE) loader: validate the MZ header, compute the load
 * module layout (header size, load image, relocation table), parse the
 * relocation entries (segment:offset fixups), and return the entry point as
 * a linear address (CS:IP) plus the load-segment base. This is the in-kernel
 * execution leg that lets a DOS executable run ON the AGI kernel.
 * Deterministic, real work (no stubs). */

#include <stddef.h>
#include <stdint.h>

typedef enum {
    WUBU_DOS_OK = 0,
    WUBU_DOS_ERR_ARGS = -1,
    WUBU_DOS_ERR_TRUNC = -2,     /* buffer too small for header */
    WUBU_DOS_ERR_MZ = -3,        /* not an MZ executable */
    WUBU_DOS_ERR_LAYOUT = -4,    /* inconsistent header (blocks/relocs) */
    WUBU_DOS_ERR_RELOC = -5,     /* relocation table overruns buffer */
    WUBU_DOS_ERR_ARENA = -6      /* arena too small for load image */
} wubu_dos_status_t;

typedef struct {
    uint16_t header_paras;   /* header size in 16-byte paragraphs (e_cparhdr) */
    uint16_t ip;             /* entry IP */
    uint16_t cs;             /* entry CS */
    uint32_t load_offset;    /* byte offset of the load module in the file */
    uint32_t load_size;      /* size of the load module in bytes */
    uint16_t min_alloc;      /* e_minalloc (extra paragraphs below load) */
    uint16_t max_alloc;      /* e_maxalloc */
    uint16_t ss;             /* stack segment (relative to load) */
    uint16_t sp;             /* stack pointer */
    uint16_t n_relocs;       /* number of relocation entries */
    uint32_t entry_linear;   /* entry as linear addr (load_seg*16 + ip) */
    uint32_t load_seg;       /* load segment base (paragraph) chosen for it */
} wubu_dos_image_t;

/* Parse+validate an MZ executable from `buf` (length `len`) and compute its
 * load layout into `img` (no bytes are copied). `load_seg` is the paragraph
 * the loader would place the image at; entry_linear is computed from it.
 * Returns WUBU_DOS_OK or a negative status. */
int wubu_dos_parse(const uint8_t *buf, size_t len, uint32_t load_seg,
                   wubu_dos_image_t *img);

/* Copy the load module of a parsed image into `arena` (capacity `arena_cap`).
 * The caller must have parsed first. Returns WUBU_DOS_OK or negative. */
int wubu_dos_load_image(const uint8_t *buf, size_t len,
                        const wubu_dos_image_t *img, uint8_t *arena,
                        size_t arena_cap);

/* Apply one relocation entry (index `i`) in the loaded arena: read the
 * segment:offset word from the relocation table, add the load segment to the
 * 16-bit word at that offset in the arena. Returns 0 on success, -1 on bad
 * index/overrun. */
int wubu_dos_apply_reloc(const uint8_t *buf, size_t len,
                         const wubu_dos_image_t *img, uint8_t *arena,
                         int i);

/* Human-readable name for a status code. */
const char *wubu_dos_strerror(int status);

#endif
