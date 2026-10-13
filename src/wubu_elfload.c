/* wubu_elfload.c -- RUNTIME #343 ELF execution leg.
 * See wubu_elfload.h. Real, deterministic ELF64 LE x86-64 loader. */

#include "wubu_elfload.h"
#include <string.h>

/* -- ELF64 structures (little-endian, x86-64) ---------------------------- */
#define EI_NIDENT 16
#define ELFMAG0 0x7f
#define ELFMAG1 'E'
#define ELFMAG2 'L'
#define ELFMAG3 'F'
#define ELFCLASS64 2
#define ELFDATA2LSB 1
#define EM_X86_64 62
#define ET_EXEC 2
#define ET_DYN 3
#define PT_LOAD 1
#define PT_INTERP 3
#define PF_X 1
#define PF_W 2
#define PF_R 4

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const uint8_t *p) {
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

const char *wubu_elf_strerror(int status)
{
    switch (status) {
    case WUBU_ELF_OK:          return "ok";
    case WUBU_ELF_ERR_ARGS:    return "bad arguments";
    case WUBU_ELF_ERR_TRUNC:   return "truncated buffer";
    case WUBU_ELF_ERR_MAGIC:   return "not an ELF (bad magic)";
    case WUBU_ELF_ERR_CLASS:   return "not ELFCLASS64";
    case WUBU_ELF_ERR_ENDIAN:  return "not little-endian";
    case WUBU_ELF_ERR_MACHINE: return "not EM_X86_64";
    case WUBU_ELF_ERR_TYPE:    return "not ET_EXEC/ET_DYN";
    case WUBU_ELF_ERR_NOENTRY: return "no PT_LOAD segments";
    case WUBU_ELF_ERR_ARENA:   return "arena too small";
    case WUBU_ELF_ERR_SEGMENT: return "bad program header";
    default:                   return "unknown";
    }
}

int wubu_elf_validate(const uint8_t *buf, size_t len)
{
    if (!buf) return WUBU_ELF_ERR_ARGS;
    if (len < 64) return WUBU_ELF_ERR_TRUNC;   /* e_ehsize for ELF64 = 64 */
    if (buf[0] != ELFMAG0 || buf[1] != ELFMAG1 ||
        buf[2] != ELFMAG2 || buf[3] != ELFMAG3)
        return WUBU_ELF_ERR_MAGIC;
    if (buf[4] != ELFCLASS64) return WUBU_ELF_ERR_CLASS;
    if (buf[5] != ELFDATA2LSB) return WUBU_ELF_ERR_ENDIAN;
    if (rd16(buf + 18) != EM_X86_64) return WUBU_ELF_ERR_MACHINE;
    uint16_t etype = rd16(buf + 16);
    if (etype != ET_EXEC && etype != ET_DYN) return WUBU_ELF_ERR_TYPE;
    return WUBU_ELF_OK;
}

int wubu_elf_load(const uint8_t *buf, size_t len,
                  uint8_t *arena, size_t arena_cap, uint64_t base,
                  wubu_elf_image_t *img)
{
    if (!buf || !arena || !img) return WUBU_ELF_ERR_ARGS;
    int v = wubu_elf_validate(buf, len);
    if (v != WUBU_ELF_OK) return v;

    uint64_t e_entry   = rd64(buf + 24);
    uint64_t e_phoff   = rd64(buf + 32);
    uint16_t e_phentsize = rd16(buf + 54);
    uint16_t e_phnum   = rd16(buf + 56);

    if (e_phentsize < 56) return WUBU_ELF_ERR_SEGMENT;  /* ELF64 phdr = 56B */
    if (e_phnum == 0) return WUBU_ELF_ERR_NOENTRY;
    /* program-header table must fit in the buffer */
    if (e_phoff > len || (uint64_t)e_phnum * e_phentsize > len - e_phoff)
        return WUBU_ELF_ERR_TRUNC;

    /* First pass: determine the image extent so we can bounds-check the
     * arena once, then map. */
    uint64_t lo = (uint64_t)-1, hi = 0;
    int n_load = 0, has_interp = 0;
    for (int i = 0; i < e_phnum; i++) {
        const uint8_t *ph = buf + e_phoff + (uint64_t)i * e_phentsize;
        uint32_t p_type = rd32(ph);
        if (p_type == PT_INTERP) { has_interp = 1; continue; }
        if (p_type != PT_LOAD) continue;
        uint64_t p_vaddr  = rd64(ph + 16);
        uint64_t p_memsz  = rd64(ph + 40);
        if (p_memsz == 0) continue;
        uint64_t s_lo = p_vaddr & ~(uint64_t)(WUBU_ELF_PAGE - 1);
        uint64_t s_hi = (p_vaddr + p_memsz + WUBU_ELF_PAGE - 1) &
                        ~(uint64_t)(WUBU_ELF_PAGE - 1);
        if (s_lo < lo) lo = s_lo;
        if (s_hi > hi) hi = s_hi;
        n_load++;
    }
    if (n_load == 0) return WUBU_ELF_ERR_NOENTRY;

    uint64_t image_size = hi - lo;
    if (image_size > arena_cap) return WUBU_ELF_ERR_ARENA;
    /* p_offset/p_filesz sanity is checked per-segment during mapping. */

    memset(arena, 0, (size_t)image_size);

    /* Second pass: copy file bytes, zero the rest (BSS). */
    for (int i = 0; i < e_phnum; i++) {
        const uint8_t *ph = buf + e_phoff + (uint64_t)i * e_phentsize;
        uint32_t p_type = rd32(ph);
        if (p_type != PT_LOAD) continue;
        uint64_t p_offset = rd64(ph + 8);
        uint64_t p_vaddr  = rd64(ph + 16);
        uint64_t p_filesz = rd64(ph + 32);
        uint64_t p_memsz  = rd64(ph + 40);
        if (p_memsz == 0) continue;
        if (p_filesz > p_memsz) return WUBU_ELF_ERR_SEGMENT;
        /* file bytes must lie within the input buffer */
        if (p_offset > len || p_filesz > len - p_offset)
            return WUBU_ELF_ERR_TRUNC;
        uint64_t dst = p_vaddr - lo;
        if (dst + p_memsz > image_size) return WUBU_ELF_ERR_ARENA;
        if (p_filesz) memcpy(arena + dst, buf + p_offset, (size_t)p_filesz);
        /* remainder [p_filesz, p_memsz) already zeroed by the memset above */
    }

    img->base = base;
    img->entry = base + e_entry;
    img->min_vaddr = lo;
    img->max_vaddr = hi;
    img->image_size = image_size;
    img->n_load = n_load;
    img->has_interp = has_interp;
    img->phnum = e_phnum;
    img->phoff = e_phoff;
    img->phentsize = e_phentsize;
    return WUBU_ELF_OK;
}
