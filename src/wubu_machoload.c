/* wubu_machoload.c -- RUNTIME #347 Mach-O execution leg.
 * See wubu_machoload.h. Real, deterministic Mach-O 64 x86_64 loader. */

#include "wubu_machoload.h"
#include <string.h>

/* -- Mach-O 64 constants (little-endian, x86_64) ------------------------- */
#define MH_MAGIC_64   0xfeedfacf
#define CPU_TYPE_X86_64 0x01000007
#define CPU_ARCH_ABI64 0x01000000
#define MH_EXECUTE    2
#define MH_DYLIB      6
#define MH_BUNDLE     8
#define LC_SEGMENT_64 0x19

static uint16_t m_rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t m_rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t m_rd64(const uint8_t *p) {
    return (uint64_t)m_rd32(p) | ((uint64_t)m_rd32(p + 4) << 32);
}

const char *wubu_macho_strerror(int status)
{
    switch (status) {
    case WUBU_MACHO_OK:          return "ok";
    case WUBU_MACHO_ERR_ARGS:    return "bad arguments";
    case WUBU_MACHO_ERR_TRUNC:   return "truncated buffer";
    case WUBU_MACHO_ERR_MAGIC:   return "not a Mach-O (bad magic)";
    case WUBU_MACHO_ERR_CLASS:   return "not 64-bit Mach-O";
    case WUBU_MACHO_ERR_ENDIAN:  return "not little-endian";
    case WUBU_MACHO_ERR_CPU:     return "not x86_64";
    case WUBU_MACHO_ERR_FILETYPE:return "unsupported filetype";
    case WUBU_MACHO_ERR_CMDS:    return "load commands overrun buffer";
    case WUBU_MACHO_ERR_NOENTRY: return "no mappable segments";
    case WUBU_MACHO_ERR_ARENA:   return "arena too small";
    case WUBU_MACHO_ERR_SEGMENT: return "bad segment layout";
    default:                     return "unknown";
    }
}

int wubu_macho_validate(const uint8_t *buf, size_t len)
{
    if (!buf) return WUBU_MACHO_ERR_ARGS;
    if (len < 32) return WUBU_MACHO_ERR_TRUNC;   /* mach_header_64 = 32 bytes */
    if (m_rd32(buf) != MH_MAGIC_64) {
        /* big-endian Mach-O magic is 0xcffaedfe when read LE here */
        return WUBU_MACHO_ERR_MAGIC;
    }
    /* magic fedfacf read LE means bytes cf fa ed fe => host is LE; correct. */
    uint32_t cputype = m_rd32(buf + 4);
    if (cputype != CPU_TYPE_X86_64) return WUBU_MACHO_ERR_CPU;
    uint32_t filetype = m_rd32(buf + 12);
    if (filetype != MH_EXECUTE && filetype != MH_DYLIB && filetype != MH_BUNDLE)
        return WUBU_MACHO_ERR_FILETYPE;
    return WUBU_MACHO_OK;
}

int wubu_macho_load(const uint8_t *buf, size_t len,
                    uint8_t *arena, size_t arena_cap, uint64_t base,
                    wubu_macho_image_t *img)
{
    if (!buf || !arena || !img) return WUBU_MACHO_ERR_ARGS;
    int v = wubu_macho_validate(buf, len);
    if (v != WUBU_MACHO_OK) return v;

    uint32_t ncmds = m_rd32(buf + 16);
    uint32_t sizeofcmds = m_rd32(buf + 20);
    uint64_t entryoff = 0;

    /* the load-command block starts right after the 32-byte header */
    if (sizeofcmds > len - 32) return WUBU_MACHO_ERR_CMDS;

    /* First pass: image extent over LC_SEGMENT_64 segments. */
    uint64_t lo = (uint64_t)-1, hi = 0;
    int n_seg = 0;
    uint64_t off = 32;
    for (uint32_t i = 0; i < ncmds; i++) {
        if (off + 8 > 32 + sizeofcmds) return WUBU_MACHO_ERR_CMDS;
        const uint8_t *lc = buf + off;
        uint32_t cmd = m_rd32(lc);
        uint32_t cmdsize = m_rd32(lc + 4);
        if (cmdsize < 8 || off + cmdsize > 32 + sizeofcmds)
            return WUBU_MACHO_ERR_CMDS;
        if (cmd == LC_SEGMENT_64) {
            if (cmdsize < 72) return WUBU_MACHO_ERR_SEGMENT;
            uint64_t vmaddr = m_rd64(lc + 24);
            uint64_t vmsize = m_rd64(lc + 32);
            if (vmsize == 0) { off += cmdsize; continue; }
            uint64_t s_lo = vmaddr & ~(uint64_t)(WUBU_MACHO_PAGE - 1);
            uint64_t s_hi = (vmaddr + vmsize + WUBU_MACHO_PAGE - 1) &
                            ~(uint64_t)(WUBU_MACHO_PAGE - 1);
            if (s_lo < lo) lo = s_lo;
            if (s_hi > hi) hi = s_hi;
            n_seg++;
        }
        off += cmdsize;
    }
    if (n_seg == 0) return WUBU_MACHO_ERR_NOENTRY;

    uint64_t image_size = hi - lo;
    if (image_size > arena_cap) return WUBU_MACHO_ERR_ARENA;
    memset(arena, 0, (size_t)image_size);

    /* Second pass: copy segment file bytes; remainder is zero (BSS). */
    off = 32;
    for (uint32_t i = 0; i < ncmds; i++) {
        const uint8_t *lc = buf + off;
        uint32_t cmd = m_rd32(lc);
        uint32_t cmdsize = m_rd32(lc + 4);
        if (cmd == LC_SEGMENT_64) {
            uint64_t vmaddr   = m_rd64(lc + 24);
            uint64_t vmsize   = m_rd64(lc + 32);
            uint64_t fileoff  = m_rd64(lc + 40);
            uint64_t filesize = m_rd64(lc + 48);
            if (vmsize > 0) {
                if (filesize > vmsize) return WUBU_MACHO_ERR_SEGMENT;
                if (fileoff > len || filesize > len - fileoff)
                    return WUBU_MACHO_ERR_TRUNC;
                uint64_t dst = vmaddr - lo;
                if (dst + vmsize > image_size) return WUBU_MACHO_ERR_ARENA;
                if (filesize) memcpy(arena + dst, buf + fileoff, (size_t)filesize);
            }
        }
        off += cmdsize;
    }

    /* entry: for LC_UNIXTHREAD/LC_MAIN we do not decode the thread state;
     * use the lowest segment vmaddr (the __TEXT start, conventional _start
     * location) as the entry offset. arena[0] maps to vmaddr `lo`, and `base`
     * is that arena slot's absolute address, so the absolute entry is
     * base + (entryoff - lo). */
    entryoff = lo;

    img->base = base;
    img->entry = base + (entryoff - lo);
    img->min_vmaddr = lo;
    img->max_vmaddr = hi;
    img->image_size = image_size;
    img->n_seg = n_seg;
    img->filetype = (int)m_rd32(buf + 12);
    img->cputype = m_rd32(buf + 4);
    img->ncmds = ncmds;
    return WUBU_MACHO_OK;
}
