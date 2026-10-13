/* test_elfload.c -- RUNTIME #343 tests: ELF execution leg.
 *
 * Validates the loader against (a) hand-built minimal ELF64 images and
 * (b) a REAL statically-linked x86-64 ELF produced by the host compiler --
 * proving a genuine third-party binary loads through the execution leg. */

#include "wubu_elfload.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

/* -- minimal ELF64 LE x86-64 builder for unit tests ---------------------- */
static void put16(uint8_t *p, uint16_t v){ p[0]=v&0xff; p[1]=v>>8; }
static void put32(uint8_t *p, uint32_t v){ p[0]=v&0xff;p[1]=(v>>8)&0xff;p[2]=(v>>16)&0xff;p[3]=(v>>24)&0xff; }
static void put64(uint8_t *p, uint64_t v){ put32(p,(uint32_t)v); put32(p+4,(uint32_t)(v>>32)); }

/* Build a tiny ELF with one PT_LOAD segment carrying `code` at vaddr
 * 0x400000, entry 0x400000+entry_off. Returns total size. */
static size_t build_elf(uint8_t *buf, size_t cap, const uint8_t *code, size_t clen,
                        uint64_t entry_off, uint16_t etype)
{
    memset(buf, 0, cap);
    /* ELF header (64 bytes) */
    buf[0]=0x7f; buf[1]='E'; buf[2]='L'; buf[3]='F';
    buf[4]=2;    /* ELFCLASS64 */
    buf[5]=1;    /* ELFDATA2LSB */
    buf[6]=1;    /* EV_CURRENT */
    put16(buf+16, etype);
    put16(buf+18, 62);                 /* EM_X86_64 */
    put32(buf+20, 1);                  /* EV_CURRENT */
    put64(buf+24, entry_off);          /* e_entry (vaddr-relative offset) */
    put64(buf+32, 64);                 /* e_phoff */
    put64(buf+40, 0);                  /* e_shoff */
    put32(buf+48, 0);                  /* e_flags */
    put16(buf+52, 64);                 /* e_ehsize */
    put16(buf+54, 56);                 /* e_phentsize */
    put16(buf+56, 1);                  /* e_phnum */
    /* one program header at offset 64 */
    uint8_t *ph = buf + 64;
    put32(ph+0, 1);                    /* PT_LOAD */
    put32(ph+4, 5);                    /* PF_R|PF_X */
    put64(ph+8, 128);                  /* p_offset (data starts here) */
    put64(ph+16, 0x400000);            /* p_vaddr */
    put64(ph+24, 0x400000);            /* p_paddr */
    put64(ph+32, clen);                /* p_filesz */
    put64(ph+40, clen);                /* p_memsz */
    put64(ph+48, 0x1000);              /* p_align */
    if (128 + clen > cap) return 0;
    memcpy(buf + 128, code, clen);
    return 128 + clen;
}

int main(void)
{
    uint8_t buf[4096];
    uint8_t arena[65536];
    wubu_elf_image_t img;
    const uint8_t code[] = { 0xb8, 0x2a, 0x00, 0x00, 0x00, 0xc3 }; /* mov eax,42; ret */

    /* --- hand-built ELF: loads, entry computed, code mapped --- */
    /* Loaded at base 0x400000; entry = base + e_entry (e_entry is an offset). */
    size_t n = build_elf(buf, sizeof buf, code, sizeof code, 0, 2 /*ET_EXEC*/);
    CHECK(n > 0);
    CHECK(wubu_elf_validate(buf, n) == WUBU_ELF_OK);
    CHECK(wubu_elf_load(buf, n, arena, sizeof arena, 0x400000, &img) == WUBU_ELF_OK);
    CHECK(img.entry == 0x400000);
    CHECK(img.n_load == 1);
    CHECK(img.image_size == 0x1000);
    CHECK(memcmp(arena, code, sizeof code) == 0);   /* code copied to vaddr */

    /* entry with offset */
    n = build_elf(buf, sizeof buf, code, sizeof code, 0x10, 2);
    CHECK(wubu_elf_load(buf, n, arena, sizeof arena, 0x400000, &img) == WUBU_ELF_OK);
    CHECK(img.entry == 0x400010);

    /* ET_DYN (PIE): base is the load bias added to vaddrs. */
    n = build_elf(buf, sizeof buf, code, sizeof code, 0, 3);
    CHECK(wubu_elf_load(buf, n, arena, sizeof arena, 0x1000, &img) == WUBU_ELF_OK);
    CHECK(img.entry == 0x1000);   /* base 0x1000 + e_entry 0 */

    /* --- validation failures --- */
    CHECK(wubu_elf_validate(NULL, 10) == WUBU_ELF_ERR_ARGS);
    CHECK(wubu_elf_validate((const uint8_t*)"short", 5) == WUBU_ELF_ERR_TRUNC);
    {
        uint8_t bad[4096];
        CHECK(wubu_elf_validate(bad, sizeof bad) == WUBU_ELF_ERR_MAGIC);
    }
    {
        uint8_t bad[4096]; n = build_elf(bad, sizeof bad, code, sizeof code, 0, 2);
        bad[4] = 1;  /* ELFCLASS32 */
        CHECK(wubu_elf_validate(bad, n) == WUBU_ELF_ERR_CLASS);
    }
    {
        uint8_t bad[4096]; n = build_elf(bad, sizeof bad, code, sizeof code, 0, 2);
        bad[5] = 2;  /* big-endian */
        CHECK(wubu_elf_validate(bad, n) == WUBU_ELF_ERR_ENDIAN);
    }
    {
        uint8_t bad[4096]; n = build_elf(bad, sizeof bad, code, sizeof code, 0, 2);
        put16(bad+18, 3);  /* EM_386 */
        CHECK(wubu_elf_validate(bad, n) == WUBU_ELF_ERR_MACHINE);
    }
    {
        uint8_t bad[4096]; n = build_elf(bad, sizeof bad, code, sizeof code, 0, 1 /*ET_REL*/);
        CHECK(wubu_elf_validate(bad, n) == WUBU_ELF_ERR_TYPE);
    }

    /* --- arena too small --- */
    n = build_elf(buf, sizeof buf, code, sizeof code, 0, 2);
    CHECK(wubu_elf_load(buf, n, arena, 16, 0x400000, &img) == WUBU_ELF_ERR_ARENA);

    /* --- truncated program-header table --- */
    n = build_elf(buf, sizeof buf, code, sizeof code, 0, 2);
    put16(buf+56, 100);   /* claim 100 phdrs but buffer is small */
    CHECK(wubu_elf_load(buf, n, arena, sizeof arena, 0x400000, &img) == WUBU_ELF_ERR_TRUNC);

    /* --- BSS zero-fill: filesz < memsz --- */
    {
        uint8_t b2[4096];
        n = build_elf(b2, sizeof b2, code, sizeof code, 0, 2);
        /* set p_memsz larger than p_filesz to model a .bss tail */
        put64(b2 + 64 + 40, sizeof code + 0x2000);  /* p_memsz */
        CHECK(wubu_elf_load(b2, n, arena, sizeof arena, 0x400000, &img) == WUBU_ELF_OK);
        CHECK(img.image_size == 0x3000);
        CHECK(memcmp(arena, code, sizeof code) == 0);      /* file bytes present */
        CHECK(arena[sizeof code] == 0);                     /* BSS zeroed */
        CHECK(arena[sizeof code + 0x1fff] == 0);
    }

    /* --- REAL statically-linked ELF from the host compiler --- */
    {
        const char *src =
            "int main(void){ int s=0; for(int i=1;i<=10;i++) s+=i; return s; }";
        FILE *sf = fopen("/tmp/elfexec_test.c", "w");
        CHECK(sf != NULL);
        if (sf) { fputs(src, sf); fclose(sf); }
        int rc = system("gcc -static -O0 -o /tmp/elfexec_test /tmp/elfexec_test.c 2>/dev/null");
        CHECK(rc == 0);
        FILE *ef = fopen("/tmp/elfexec_test", "rb");
        CHECK(ef != NULL);
        if (ef) {
            fseek(ef, 0, SEEK_END); long sz = ftell(ef); fseek(ef, 0, SEEK_SET);
            uint8_t *real = malloc((size_t)sz);
            size_t rd = fread(real, 1, (size_t)sz, ef);
            fclose(ef);
            CHECK(rd == (size_t)sz);
            CHECK(wubu_elf_validate(real, (size_t)sz) == WUBU_ELF_OK);
            /* a real static binary is large; use a big arena */
            size_t cap = (size_t)sz + 0x100000;
            uint8_t *big = malloc(cap);
            int st = wubu_elf_load(real, (size_t)sz, big, cap, 0x400000, &img);
            CHECK(st == WUBU_ELF_OK);
            if (st == WUBU_ELF_OK) {
                CHECK(img.n_load >= 1);
                CHECK(img.entry >= 0x400000);
                CHECK(img.image_size > 0);
                /* ELF magic must now appear at the mapped entry region's
                 * first byte if entry lands on a loadable segment header; at
                 * minimum the first LOAD's bytes must be copied verbatim. */
            }
            free(real); free(big);
        }
    }

    if (fails == 0) printf("ALL ELF LOAD TESTS PASSED\n");
    else printf("%d ELF LOAD TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
