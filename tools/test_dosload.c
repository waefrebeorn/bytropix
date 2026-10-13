/* test_dosload.c -- RUNTIME #345 tests: DOS execution leg.
 *
 * Validates the MZ loader against hand-built executables and against a REAL
 * DOS code payload (assembled by nasm) packaged under a correct MZ header. */

#include "wubu_dosload.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

static void z_put16(uint8_t *p, uint16_t v){ p[0]=v&0xff; p[1]=v>>8; }

/* Build an MZ with header_paras=H, load payload `code` (len clen), entry IP
 * at offset E in the payload, one optional relocation (off,seg). */
static size_t build_mz(uint8_t *buf, size_t cap, uint16_t header_paras,
                       const uint8_t *code, size_t clen,
                       uint16_t ip, uint16_t cs, uint16_t ss, uint16_t sp,
                       int with_reloc, uint16_t rel_off, uint16_t rel_seg)
{
    memset(buf, 0, cap);
    buf[0]='M'; buf[1]='Z';
    uint32_t load_off = (uint32_t)header_paras * 16;
    /* load size rounded up to 512-byte page */
    uint32_t pages = (uint32_t)((clen + 511) / 512);
    if (pages == 0) pages = 1;
    uint16_t cblp = (uint16_t)(clen % 512);   /* bytes in last page (0 => full) */
    z_put16(buf + 2, cblp);       /* e_cblp */
    z_put16(buf + 4, (uint16_t)pages); /* e_cp */
    uint16_t nreloc = with_reloc ? 1 : 0;
    z_put16(buf + 6, nreloc);     /* e_crlc */
    z_put16(buf + 8, header_paras); /* e_cparhdr */
    z_put16(buf + 10, 0x10);      /* e_minalloc */
    z_put16(buf + 12, 0xffff);    /* e_maxalloc */
    z_put16(buf + 14, ss);
    z_put16(buf + 16, sp);
    z_put16(buf + 20, ip);        /* e_ip */
    z_put16(buf + 22, cs);        /* e_cs */
    if (with_reloc) {
        z_put16(buf + 24, (uint16_t)load_off - 4); /* e_lfarlc: reloc before code */
        uint8_t *r = buf + load_off - 4;
        z_put16(r, rel_off);
        z_put16(r + 2, rel_seg);
    } else {
        z_put16(buf + 24, (uint16_t)load_off);     /* e_lfarlc = load start */
    }
    if (load_off + clen > cap) return 0;
    memcpy(buf + load_off, code, clen);
    return load_off + clen;
}

int main(void)
{
    uint8_t buf[8192];
    uint8_t arena[8192];
    wubu_dos_image_t img;
    /* 6-byte payload: mov ax,0x4c00 (B8 00 4C); int 0x21 (CD 21) */
    const uint8_t code[] = { 0xB8, 0x00, 0x4C, 0xCD, 0x21, 0x00 };

    /* --- hand-built MZ: parse, entry, image copy --- */
    size_t n = build_mz(buf, sizeof buf, 4, code, sizeof code, 0, 0, 0, 0x100, 0, 0, 0);
    CHECK(n > 0);
    CHECK(wubu_dos_parse(buf, n, 0x2000, &img) == WUBU_DOS_OK);
    CHECK(img.header_paras == 4);
    CHECK(img.load_offset == 64);            /* 4 paras * 16 */
    CHECK(img.ip == 0);
    CHECK(img.entry_linear == 0x2000 * 16 + 0);  /* load_seg*16 + ip */
    CHECK(img.n_relocs == 0);
    CHECK(wubu_dos_load_image(buf, n, &img, arena, sizeof arena) == WUBU_DOS_OK);
    CHECK(memcmp(arena, code, sizeof code) == 0);   /* payload copied */

    /* entry IP nonzero */
    n = build_mz(buf, sizeof buf, 4, code, sizeof code, 0x10, 0, 0, 0x100, 0, 0, 0);
    CHECK(wubu_dos_parse(buf, n, 0x1000, &img) == WUBU_DOS_OK);
    CHECK(img.entry_linear == 0x1000 * 16 + 0x10);

    /* --- validation failures --- */
    CHECK(wubu_dos_parse(NULL, 100, 0x1000, &img) == WUBU_DOS_ERR_ARGS);
    CHECK(wubu_dos_parse((const uint8_t*)"tiny", 4, 0x1000, &img) == WUBU_DOS_ERR_TRUNC);
    {
        uint8_t bad[28]; memset(bad, 0, sizeof bad);
        CHECK(wubu_dos_parse(bad, sizeof bad, 0x1000, &img) == WUBU_DOS_ERR_MZ);
    }
    /* header_paras=0 -> layout error (build valid, then zero cparhdr) */
    n = build_mz(buf, sizeof buf, 4, code, sizeof code, 0, 0, 0, 0x100, 0, 0, 0);
    z_put16(buf + 8, 0);   /* e_cparhdr = 0 */
    CHECK(wubu_dos_parse(buf, n, 0x1000, &img) == WUBU_DOS_ERR_LAYOUT);

    /* --- arena too small --- */
    n = build_mz(buf, sizeof buf, 4, code, sizeof code, 0, 0, 0, 0x100, 0, 0, 0);
    CHECK(wubu_dos_parse(buf, n, 0x1000, &img) == WUBU_DOS_OK);
    CHECK(wubu_dos_load_image(buf, n, &img, arena, 2) == WUBU_DOS_ERR_ARENA);

    /* --- relocation apply --- */
    {
        /* payload with a placeholder word at offset 2 that should be fixed up */
        uint8_t rc[8] = { 0xB8, 0x34, 0x12, 0xCD, 0x21, 0, 0, 0 };
        n = build_mz(buf, sizeof buf, 4, rc, sizeof rc, 0, 0, 0, 0x100, 1, 1, 0);
        CHECK(wubu_dos_parse(buf, n, 0x3000, &img) == WUBU_DOS_OK);
        CHECK(img.n_relocs == 1);
        CHECK(wubu_dos_load_image(buf, n, &img, arena, sizeof arena) == WUBU_DOS_OK);
        /* word at arena[1..2] = 0x1234; after reloc +load_seg(0x3000) */
        uint16_t before = (uint16_t)(arena[1] | (arena[2] << 8));
        CHECK(before == 0x1234);
        CHECK(wubu_dos_apply_reloc(buf, n, &img, arena, 0) == 0);
        uint16_t after = (uint16_t)(arena[1] | (arena[2] << 8));
        CHECK(after == (uint16_t)(0x1234 + 0x3000));
        /* out-of-range reloc index */
        CHECK(wubu_dos_apply_reloc(buf, n, &img, arena, 5) == -1);
    }

    /* --- REAL nasm-assembled DOS code under a correct MZ header --- */
    {
        int rc = system("printf 'bits 16\\norg 0x0\\nstart:\\n mov ax,0x4c00\\n int 0x21\\n' "
                        "> /tmp/dostest.asm && nasm -f bin -o /tmp/dos_payload.bin /tmp/dostest.asm 2>/dev/null");
        CHECK(rc == 0);
        FILE *df = fopen("/tmp/dos_payload.bin", "rb");
        CHECK(df != NULL);
        if (df) {
            fseek(df, 0, SEEK_END); long sz = ftell(df); fseek(df, 0, SEEK_SET);
            uint8_t *payload = malloc((size_t)(sz > 0 ? sz : 1));
            size_t rd = fread(payload, 1, (size_t)sz, df);
            fclose(df);
            CHECK(rd == (size_t)sz && sz > 0);
            /* package the real code under a valid MZ header */
            uint8_t *full = malloc((size_t)sz + 256);
            size_t fn = build_mz(full, (size_t)sz + 256, 4, payload, (size_t)sz,
                                 0, 0, 0, 0x100, 0, 0, 0);
            CHECK(fn > 0);
            CHECK(wubu_dos_parse(full, fn, 0x5000, &img) == WUBU_DOS_OK);
            CHECK(img.load_size == (uint32_t)sz);
            CHECK(wubu_dos_load_image(full, fn, &img, arena, sizeof arena) == WUBU_DOS_OK);
            CHECK(memcmp(arena, payload, (size_t)sz) == 0);   /* real code loaded */
            CHECK(img.entry_linear == 0x5000 * 16);
            free(payload); free(full);
        }
    }

    if (fails == 0) printf("ALL DOS LOAD TESTS PASSED\n");
    else printf("%d DOS LOAD TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
