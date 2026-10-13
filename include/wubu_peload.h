#ifndef WUBU_PELOAD_H
#define WUBU_PELOAD_H

/* wubu_peload.h -- RUNTIME #344 PE execution leg.
 *
 * A minimal PE/PE32+ loader: validate the DOS header (MZ) and PE signature,
 * read the COFF + optional header, and map every section into a page-aligned
 * arena at its VirtualAddress (relative to ImageBase), honouring
 * SizeOfRawData/VirtualSize (zero-fill the remainder as BSS). Returns the
 * computed entry point (ImageBase + AddressOfEntryPoint). This is the
 * in-kernel execution leg that lets a Windows PE binary run ON the AGI
 * kernel. Deterministic, real work (no stubs). */

#include <stddef.h>
#include <stdint.h>

#define WUBU_PE_PAGE 4096

typedef enum {
    WUBU_PE_OK = 0,
    WUBU_PE_ERR_ARGS = -1,
    WUBU_PE_ERR_TRUNC = -2,
    WUBU_PE_ERR_MZ = -3,          /* not a DOS MZ executable */
    WUBU_PE_ERR_PE = -4,          /* bad PE signature */
    WUBU_PE_ERR_MACHINE = -5,     /* not IMAGE_FILE_MACHINE_AMD64 */
    WUBU_PE_ERR_OPTMAGIC = -6,    /* not PE32+ (0x20b) */
    WUBU_PE_ERR_NOSEC = -7,       /* no sections */
    WUBU_PE_ERR_ARENA = -8,       /* arena too small */
    WUBU_PE_ERR_SECTION = -9      /* bad section layout */
} wubu_pe_status_t;

typedef struct {
    uint64_t image_base;     /* preferred load base (ImageBase) */
    uint64_t entry;          /* absolute entry (ImageBase + AddressOfEntryPoint) */
    uint64_t min_rva;        /* lowest mapped RVA (page-aligned) */
    uint64_t max_rva;        /* one past the highest mapped RVA */
    uint64_t image_size;     /* max_rva - min_rva */
    int n_sections;          /* number of sections mapped */
    uint16_t machine;
    uint16_t opt_magic;
} wubu_pe_image_t;

/* Parse+load a PE32+ image from `buf` (length `len`) into `arena` (capacity
 * `arena_cap`) mapped at base `base`. On success fills `img`, returns
 * WUBU_PE_OK; on failure returns a negative status, leaving arena/img
 * untouched. */
int wubu_pe_load(const uint8_t *buf, size_t len,
                 uint8_t *arena, size_t arena_cap, uint64_t base,
                 wubu_pe_image_t *img);

/* Validate a PE32+ header without loading. */
int wubu_pe_validate(const uint8_t *buf, size_t len);

/* Human-readable name for a status code. */
const char *wubu_pe_strerror(int status);

#endif
