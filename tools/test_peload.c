/* test_peload.c -- RUNTIME #344 tests: PE execution leg.
 *
 * Validates the loader against a hand-built PE32+ executable and against a
 * REAL MinGW-produced Windows PE -- the execution leg loads genuine Windows
 * binaries. */

#include "wubu_peload.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

static void p_put16(uint8_t *p, uint16_t v){ p[0]=v&0xff; p[1]=v>>8; }
static void p_put32(uint8_t *p, uint32_t v){ p[0]=v&0xff;p[1]=(v>>8)&0xff;p[2]=(v>>16)&0xff;p[3]=(v>>24)&0xff; }
static void p_put64(uint8_t *p, uint64_t v){ p_put32(p,(uint32_t)v); p_put32(p+4,(uint32_t)(v>>32)); }

/* Build a minimal PE32+ (AMD64) with one section carrying `data` at RVA
 * 0x1000, entry RVA 0x1000, ImageBase 0x140000000. Returns total size. */
static size_t build_pe(uint8_t *buf, size_t cap, const uint8_t *data, size_t dlen,
                       uint32_t entry_rva, uint16_t machine)
{
    memset(buf, 0, cap);
    const uint32_t opt_size = 240;                 /* PE32+ optional hdr */
    const uint32_t e_lfanew = 0x80;
    const uint64_t data_off = e_lfanew + 24 + opt_size + 40; /* + 1 section hdr */
    /* DOS header */
    buf[0]='M'; buf[1]='Z';
    p_put32(buf + 0x3c, e_lfanew);
    /* PE signature */
    uint8_t *pe = buf + e_lfanew;
    pe[0]='P'; pe[1]='E'; pe[2]=0; pe[3]=0;
    p_put16(pe + 4, machine);      /* machine */
    p_put16(pe + 6, 1);            /* NumberOfSections */
    p_put16(pe + 20, opt_size);    /* SizeOfOptionalHeader */
    /* optional header (PE32+) */
    uint8_t *opt = pe + 24;
    p_put16(opt + 0, 0x20b);       /* PE32+ magic */
    p_put32(opt + 16, entry_rva);  /* AddressOfEntryPoint */
    p_put64(opt + 24, 0x140000000ULL); /* ImageBase */
    /* section header */
    uint8_t *sh = pe + 24 + opt_size;
    memcpy(sh + 0, ".text", 5);
    p_put32(sh + 8, (uint32_t)dlen);   /* VirtualSize */
    p_put32(sh + 12, 0x1000);          /* VirtualAddress (RVA) */
    p_put32(sh + 16, (uint32_t)dlen);  /* SizeOfRawData */
    p_put32(sh + 20, (uint32_t)data_off); /* PointerToRawData */
    if (data_off + dlen > cap) return 0;
    memcpy(buf + data_off, data, dlen);
    return (size_t)data_off + dlen;
}

int main(void)
{
    uint8_t buf[8192];
    uint8_t arena[65536];
    wubu_pe_image_t img;
    const uint8_t data[] = { 0xb8, 0x2a, 0x00, 0x00, 0x00, 0xc3 }; /* mov eax,42; ret */

    /* --- hand-built PE32+: loads, entry computed, section mapped --- */
    size_t n = build_pe(buf, sizeof buf, data, sizeof data, 0x1000,
                        0x8664 /*IMAGE_FILE_MACHINE_AMD64*/);
    CHECK(n > 0);
    CHECK(wubu_pe_validate(buf, n) == WUBU_PE_OK);
    CHECK(wubu_pe_load(buf, n, arena, sizeof arena, 0x140000000ULL, &img) == WUBU_PE_OK);
    CHECK(img.entry == 0x140000000ULL);  /* base + (entry_rva - lo) */
    CHECK(img.n_sections == 1);
    CHECK(img.image_size == 0x1000);
    CHECK(memcmp(arena, data, sizeof data) == 0);
    CHECK(img.image_base == 0x140000000ULL);
    CHECK(img.machine == 0x8664);
    CHECK(img.opt_magic == 0x20b);

    /* --- validation failures --- */
    CHECK(wubu_pe_validate(NULL, 100) == WUBU_PE_ERR_ARGS);
    CHECK(wubu_pe_validate((const uint8_t*)"tiny", 4) == WUBU_PE_ERR_TRUNC);
    {
        uint8_t bad[64]; memset(bad, 0, sizeof bad);
        CHECK(wubu_pe_validate(bad, sizeof bad) == WUBU_PE_ERR_MZ);
    }
    {
        uint8_t bad[8192]; n = build_pe(bad, sizeof bad, data, sizeof data, 0x1000, 0x8664);
        bad[0x80] = 'X';   /* corrupt PE signature */
        CHECK(wubu_pe_validate(bad, n) == WUBU_PE_ERR_PE);
    }
    {
        uint8_t bad[8192]; n = build_pe(bad, sizeof bad, data, sizeof data, 0x1000, 0x14c /*i386*/);
        CHECK(wubu_pe_validate(bad, n) == WUBU_PE_ERR_MACHINE);
    }
    {
        uint8_t bad[8192]; n = build_pe(bad, sizeof bad, data, sizeof data, 0x1000, 0x8664);
        p_put16(bad + 0x80 + 24, 0x10b);   /* PE32, not PE32+ */
        CHECK(wubu_pe_validate(bad, n) == WUBU_PE_ERR_OPTMAGIC);
    }

    /* --- arena too small --- */
    n = build_pe(buf, sizeof buf, data, sizeof data, 0x1000, 0x8664);
    CHECK(wubu_pe_load(buf, n, arena, 16, 0x140000000ULL, &img) == WUBU_PE_ERR_ARENA);

    /* --- BSS zero-fill: SizeOfRawData < VirtualSize --- */
    {
        uint8_t b2[8192];
        n = build_pe(b2, sizeof b2, data, sizeof data, 0x1000, 0x8664);
        /* VirtualSize larger than SizeOfRawData => BSS tail zeroed */
        p_put32(b2 + 0x80 + 24 + 240 + 8, (uint32_t)sizeof data + 0x2000); /* VirtualSize */
        CHECK(wubu_pe_load(b2, n, arena, sizeof arena, 0x140000000ULL, &img) == WUBU_PE_OK);
        CHECK(img.image_size == 0x3000);
        CHECK(memcmp(arena, data, sizeof data) == 0);
        CHECK(arena[sizeof data] == 0);   /* BSS zeroed */
    }

    /* --- REAL MinGW-produced PE --- */
    {
        int rc = system("printf 'int main(void){return 42;}\\n' > /tmp/petest.c && "
                        "x86_64-w64-mingw32-gcc -o /tmp/pe_real.exe /tmp/petest.c 2>/dev/null");
        CHECK(rc == 0);
        FILE *pf = fopen("/tmp/pe_real.exe", "rb");
        CHECK(pf != NULL);
        if (pf) {
            fseek(pf, 0, SEEK_END); long sz = ftell(pf); fseek(pf, 0, SEEK_SET);
            uint8_t *real = malloc((size_t)sz);
            size_t rd = fread(real, 1, (size_t)sz, pf);
            fclose(pf);
            CHECK(rd == (size_t)sz);
            CHECK(real[0]=='M' && real[1]=='Z');
            CHECK(wubu_pe_validate(real, (size_t)sz) == WUBU_PE_OK);
            size_t cap = (size_t)sz + 0x100000;
            uint8_t *big = malloc(cap);
            int st = wubu_pe_load(real, (size_t)sz, big, cap, 0x140000000ULL, &img);
            CHECK(st == WUBU_PE_OK);
            if (st == WUBU_PE_OK) {
                CHECK(img.n_sections >= 1);
                CHECK(img.machine == 0x8664);
                CHECK(img.opt_magic == 0x20b);
                CHECK(img.image_size > 0);
                /* entry must land inside the mapped image */
                uint64_t entry_rva = img.entry - 0x140000000ULL + img.min_rva;
                CHECK(entry_rva >= img.min_rva && entry_rva < img.max_rva);
            }
            free(real); free(big);
        }
    }

    if (fails == 0) printf("ALL PE LOAD TESTS PASSED\n");
    else printf("%d PE LOAD TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
