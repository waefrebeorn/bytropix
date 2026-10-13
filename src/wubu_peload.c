/* wubu_peload.c -- RUNTIME #344 PE execution leg.
 * See wubu_peload.h. Real, deterministic PE32+ (x86_64) loader. */

#include "wubu_peload.h"
#include <string.h>

/* -- PE constants -------------------------------------------------------- */
#define IMAGE_FILE_MACHINE_AMD64 0x8664
#define PE32PLUS_MAGIC 0x20b
#define SECHDR_SIZE 40

static uint16_t p_rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t p_rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t p_rd64(const uint8_t *p) {
    return (uint64_t)p_rd32(p) | ((uint64_t)p_rd32(p + 4) << 32);
}

const char *wubu_pe_strerror(int status)
{
    switch (status) {
    case WUBU_PE_OK:          return "ok";
    case WUBU_PE_ERR_ARGS:    return "bad arguments";
    case WUBU_PE_ERR_TRUNC:   return "truncated buffer";
    case WUBU_PE_ERR_MZ:      return "not a DOS MZ executable";
    case WUBU_PE_ERR_PE:      return "bad PE signature";
    case WUBU_PE_ERR_MACHINE: return "not IMAGE_FILE_MACHINE_AMD64";
    case WUBU_PE_ERR_OPTMAGIC:return "not PE32+";
    case WUBU_PE_ERR_NOSEC:   return "no sections";
    case WUBU_PE_ERR_ARENA:   return "arena too small";
    case WUBU_PE_ERR_SECTION: return "bad section layout";
    default:                  return "unknown";
    }
}

int wubu_pe_validate(const uint8_t *buf, size_t len)
{
    if (!buf) return WUBU_PE_ERR_ARGS;
    if (len < 64) return WUBU_PE_ERR_TRUNC;
    if (buf[0] != 'M' || buf[1] != 'Z') return WUBU_PE_ERR_MZ;
    uint32_t e_lfanew = p_rd32(buf + 0x3c);
    if ((uint64_t)e_lfanew + 4 > len) return WUBU_PE_ERR_TRUNC;
    const uint8_t *pe = buf + e_lfanew;
    if (pe[0] != 'P' || pe[1] != 'E' || pe[2] != 0 || pe[3] != 0)
        return WUBU_PE_ERR_PE;
    uint16_t machine = p_rd16(pe + 4);
    if (machine != IMAGE_FILE_MACHINE_AMD64) return WUBU_PE_ERR_MACHINE;
    /* optional header starts at pe+24; magic at its offset 0 */
    uint16_t opt_magic = p_rd16(pe + 24);
    if (opt_magic != PE32PLUS_MAGIC) return WUBU_PE_ERR_OPTMAGIC;
    return WUBU_PE_OK;
}

int wubu_pe_load(const uint8_t *buf, size_t len,
                 uint8_t *arena, size_t arena_cap, uint64_t base,
                 wubu_pe_image_t *img)
{
    if (!buf || !arena || !img) return WUBU_PE_ERR_ARGS;
    int v = wubu_pe_validate(buf, len);
    if (v != WUBU_PE_OK) return v;

    uint32_t e_lfanew = p_rd32(buf + 0x3c);
    const uint8_t *pe = buf + e_lfanew;
    uint16_t nsec = p_rd16(pe + 6);
    uint16_t opt_size = p_rd16(pe + 20);
    const uint8_t *opt = pe + 24;
    uint64_t image_base = p_rd64(opt + 24);
    uint32_t entry_rva = p_rd32(opt + 16);

    if (nsec == 0) return WUBU_PE_ERR_NOSEC;
    /* section headers follow the optional header */
    uint64_t sec_off = e_lfanew + 24 + opt_size;
    if (sec_off + (uint64_t)nsec * SECHDR_SIZE > len) return WUBU_PE_ERR_TRUNC;

    /* First pass: image extent (RVA domain) over sections with nonzero VA. */
    uint64_t lo = (uint64_t)-1, hi = 0;
    int mapped = 0;
    for (int i = 0; i < nsec; i++) {
        const uint8_t *sh = buf + sec_off + (uint64_t)i * SECHDR_SIZE;
        uint32_t vsize = p_rd32(sh + 8);
        uint32_t vaddr = p_rd32(sh + 12);
        if (vsize == 0) continue;
        uint64_t s_lo = vaddr & ~(uint64_t)(WUBU_PE_PAGE - 1);
        uint64_t s_hi = (vaddr + vsize + WUBU_PE_PAGE - 1) &
                        ~(uint64_t)(WUBU_PE_PAGE - 1);
        if (s_lo < lo) lo = s_lo;
        if (s_hi > hi) hi = s_hi;
        mapped++;
    }
    if (mapped == 0) return WUBU_PE_ERR_NOSEC;

    uint64_t image_size = hi - lo;
    if (image_size > arena_cap) return WUBU_PE_ERR_ARENA;
    memset(arena, 0, (size_t)image_size);

    /* Second pass: copy raw bytes; remainder (VirtualSize>SizeOfRawData) is
     * zero-filled as BSS. SizeOfRawData is FileAlignment-rounded and may
     * exceed VirtualSize, so clamp the copy to VirtualSize. */
    for (int i = 0; i < nsec; i++) {
        const uint8_t *sh = buf + sec_off + (uint64_t)i * SECHDR_SIZE;
        uint32_t vsize = p_rd32(sh + 8);
        uint32_t vaddr = p_rd32(sh + 12);
        uint32_t rawsz = p_rd32(sh + 16);
        uint32_t rawptr = p_rd32(sh + 20);
        if (vsize == 0) continue;
        if (rawptr > len) return WUBU_PE_ERR_TRUNC;
        if (rawsz > len - rawptr) return WUBU_PE_ERR_TRUNC;
        uint32_t copy = rawsz < vsize ? rawsz : vsize;   /* clamp to VirtualSize */
        if (copy > len - rawptr) return WUBU_PE_ERR_TRUNC;
        uint64_t dst = vaddr - lo;
        if (dst + vsize > image_size) return WUBU_PE_ERR_ARENA;
        if (copy) memcpy(arena + dst, buf + rawptr, (size_t)copy);
    }

    img->image_base = image_base;
    /* entry is an RVA from ImageBase; map it through the load base */
    img->entry = base + (entry_rva - lo);
    img->min_rva = lo;
    img->max_rva = hi;
    img->image_size = image_size;
    img->n_sections = mapped;
    img->machine = p_rd16(pe + 4);
    img->opt_magic = p_rd16(opt);
    return WUBU_PE_OK;
}
