/* wubu_dosload.c -- RUNTIME #345 DOS execution leg.
 * See wubu_dosload.h. Real, deterministic MZ (DOS .EXE) loader. */

#include "wubu_dosload.h"
#include <string.h>

#define MZ_HDR_SIZE 28
#define MZ_PARA 16

static uint16_t d_rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

const char *wubu_dos_strerror(int status)
{
    switch (status) {
    case WUBU_DOS_OK:        return "ok";
    case WUBU_DOS_ERR_ARGS:  return "bad arguments";
    case WUBU_DOS_ERR_TRUNC: return "truncated buffer";
    case WUBU_DOS_ERR_MZ:    return "not an MZ executable";
    case WUBU_DOS_ERR_LAYOUT:return "inconsistent header layout";
    case WUBU_DOS_ERR_RELOC: return "relocation table overruns buffer";
    case WUBU_DOS_ERR_ARENA: return "arena too small";
    default:                 return "unknown";
    }
}

int wubu_dos_parse(const uint8_t *buf, size_t len, uint32_t load_seg,
                   wubu_dos_image_t *img)
{
    if (!buf || !img) return WUBU_DOS_ERR_ARGS;
    if (len < MZ_HDR_SIZE) return WUBU_DOS_ERR_TRUNC;
    if (buf[0] != 'M' || buf[1] != 'Z') return WUBU_DOS_ERR_MZ;

    uint16_t cblp    = d_rd16(buf + 2);   /* bytes in last page */
    uint16_t cp      = d_rd16(buf + 4);   /* 512-byte pages */
    uint16_t crlc    = d_rd16(buf + 6);   /* relocations */
    uint16_t cparhdr = d_rd16(buf + 8);   /* header size in paragraphs */
    uint16_t minalloc= d_rd16(buf + 10);
    uint16_t maxalloc= d_rd16(buf + 12);
    uint16_t ss      = d_rd16(buf + 14);
    uint16_t sp      = d_rd16(buf + 16);
    uint16_t ip      = d_rd16(buf + 20);
    uint16_t cs      = d_rd16(buf + 22);
    uint16_t reloc   = d_rd16(buf + 24);

    /* Load-module size: cp pages * 512, minus the partial-page slack when a
     * last-page byte count is present. */
    uint32_t image_bytes = (uint32_t)cp * 512;
    if (cblp) image_bytes -= (512 - cblp);
    uint32_t load_offset = (uint32_t)cparhdr * MZ_PARA;

    if (cparhdr == 0) return WUBU_DOS_ERR_LAYOUT;
    if (load_offset >= len) return WUBU_DOS_ERR_LAYOUT;
    if (image_bytes == 0) return WUBU_DOS_ERR_LAYOUT;
    if (load_offset + image_bytes > len) return WUBU_DOS_ERR_LAYOUT;

    /* relocation table must fit in the file and lie before the load module
     * (DOS stores fixups in the header area, before the load image). */
    uint32_t reloc_off = (uint32_t)reloc;
    if (crlc > 0) {
        if (reloc_off + (uint32_t)crlc * 4 > len) return WUBU_DOS_ERR_RELOC;
        if (reloc_off + (uint32_t)crlc * 4 > load_offset) return WUBU_DOS_ERR_LAYOUT;
    }

    img->header_paras = cparhdr;
    img->ip = ip;
    img->cs = cs;
    img->load_offset = load_offset;
    img->load_size = image_bytes;
    img->min_alloc = minalloc;
    img->max_alloc = maxalloc;
    img->ss = ss;
    img->sp = sp;
    img->n_relocs = crlc;
    img->load_seg = load_seg;
    /* entry linear address = load_seg*16 + ip (CS is relative to load seg) */
    img->entry_linear = (uint32_t)load_seg * MZ_PARA + ip;
    return WUBU_DOS_OK;
}

int wubu_dos_load_image(const uint8_t *buf, size_t len,
                        const wubu_dos_image_t *img, uint8_t *arena,
                        size_t arena_cap)
{
    if (!buf || !img || !arena) return WUBU_DOS_ERR_ARGS;
    if (img->load_size > arena_cap) return WUBU_DOS_ERR_ARENA;
    if (img->load_offset + img->load_size > len) return WUBU_DOS_ERR_TRUNC;
    memcpy(arena, buf + img->load_offset, img->load_size);
    return WUBU_DOS_OK;
}

int wubu_dos_apply_reloc(const uint8_t *buf, size_t len,
                         const wubu_dos_image_t *img, uint8_t *arena,
                         int i)
{
    if (!buf || !img || !arena) return WUBU_DOS_ERR_ARGS;
    if (i < 0 || i >= img->n_relocs) return -1;
    uint32_t reloc_off = d_rd16(buf + 24);   /* e_lfarlc */
    const uint8_t *e = buf + reloc_off + (uint32_t)i * 4;
    uint16_t off = d_rd16(e);
    uint16_t seg = d_rd16(e + 2);
    (void)seg;
    if (off + 1 >= img->load_size) return -1;
    /* add the load segment to the 16-bit word at `off` in the load image */
    uint16_t old = (uint16_t)(arena[off] | (arena[off + 1] << 8));
    uint16_t nv = (uint16_t)(old + img->load_seg);
    arena[off] = (uint8_t)(nv & 0xff);
    arena[off + 1] = (uint8_t)(nv >> 8);
    return 0;
}
