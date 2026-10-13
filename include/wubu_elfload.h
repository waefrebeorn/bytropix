#ifndef WUBU_ELFLOAD_H
#define WUBU_ELFLOAD_H

/* wubu_elfload.h -- RUNTIME #343 ELF execution leg.
 *
 * A minimal ELF64 little-endian x86-64 loader: parse the ELF header, validate
 * it (magic, class, endianness, machine, e_type), and map every PT_LOAD
 * segment into a contiguous page-aligned executable arena at its p_vaddr,
 * honouring p_memsz/p_filesz (BSS zero-fill) and p_flags (R/W/X). Returns the
 * computed entry point (base + e_entry). This is the in-kernel execution leg
 * that lets a third-party ELF run ON the AGI kernel. Deterministic, real work
 * (no stubs). */

#include <stddef.h>
#include <stdint.h>

#define WUBU_ELF_PAGE 4096

typedef enum {
    WUBU_ELF_OK = 0,
    WUBU_ELF_ERR_ARGS = -1,
    WUBU_ELF_ERR_TRUNC = -2,     /* buffer too small for header/segments */
    WUBU_ELF_ERR_MAGIC = -3,     /* not an ELF */
    WUBU_ELF_ERR_CLASS = -4,     /* not ELFCLASS64 */
    WUBU_ELF_ERR_ENDIAN = -5,    /* not little-endian */
    WUBU_ELF_ERR_MACHINE = -6,   /* not EM_X86_64 */
    WUBU_ELF_ERR_TYPE = -7,      /* not ET_EXEC / ET_DYN */
    WUBU_ELF_ERR_NOENTRY = -8,   /* no loadable segments */
    WUBU_ELF_ERR_ARENA = -9,     /* arena too small to hold the image */
    WUBU_ELF_ERR_SEGMENT = -10   /* bad program-header/segment layout */
} wubu_elf_status_t;

typedef struct {
    uint64_t base;          /* arena base address the image was mapped at */
    uint64_t entry;         /* absolute entry point (base + e_entry) */
    uint64_t min_vaddr;     /* lowest mapped vaddr (page-aligned) */
    uint64_t max_vaddr;     /* one past the highest mapped byte */
    uint64_t image_size;    /* max_vaddr - min_vaddr */
    int n_load;             /* number of PT_LOAD segments mapped */
    int has_interp;         /* 1 if a PT_INTERP (dynamic) segment was seen */
    uint16_t phnum;
    uint64_t phoff;
    uint16_t phentsize;
} wubu_elf_image_t;

/* Load an ELF64 LE x86-64 image from `buf` (length `len`) into `arena`
 * (capacity `arena_cap`), mapped at vaddr `base`. On success fills `img` and
 * returns WUBU_ELF_OK; on failure returns a negative wubu_elf_status_t and
 * leaves `arena`/`img` untouched. */
int wubu_elf_load(const uint8_t *buf, size_t len,
                  uint8_t *arena, size_t arena_cap, uint64_t base,
                  wubu_elf_image_t *img);

/* Validate an ELF64 LE x86-64 header without loading. Returns WUBU_ELF_OK or
 * a negative status. */
int wubu_elf_validate(const uint8_t *buf, size_t len);

/* Human-readable name for a status code. */
const char *wubu_elf_strerror(int status);

#endif
