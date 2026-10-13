/* test_dmcrypt.c -- KERNEL #28 storage path inline AES via /n.
 * Proves: encrypt changes the block (confidentiality), decrypt round-trips
 * exactly, buffer encrypt/decrypt, different keys -> different ciphertext,
 * fail-closed on NULL/bad length. */
#include "wubu_dmcrypt.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    uint32_t key[4] = { 0x01234567, 0x89ABCDEF, 0xFEDCBA98, 0x76543210 };
    wubu_dmcrypt_t c;
    CHECK(wubu_dmc_init(&c, key) == 0, "init ok");

    /* round-trip a block */
    uint32_t v[2] = { 0x11223344, 0x55667788 };
    uint32_t orig[2] = { v[0], v[1] };
    wubu_dmc_encrypt(&c, v);
    CHECK(v[0] != orig[0] || v[1] != orig[1], "encrypt changes the block (confidentiality)");
    wubu_dmc_decrypt(&c, v);
    CHECK(v[0] == orig[0] && v[1] == orig[1], "decrypt round-trips exactly");

    /* encrypt of zero block is nonzero */
    uint32_t z[2] = { 0, 0 };
    wubu_dmc_encrypt(&c, z);
    CHECK(z[0] != 0 || z[1] != 0, "zero block encrypts to nonzero");

    /* buffer round-trip (16 bytes = 2 blocks) */
    uint8_t buf[16], save[16];
    for (int i = 0; i < 16; i++) buf[i] = (uint8_t)(i * 7 + 1);
    memcpy(save, buf, 16);
    CHECK(wubu_dmc_encrypt_buf(&c, buf, 16) == 0, "encrypt buf ok");
    CHECK(memcmp(buf, save, 16) != 0, "buffer ciphertext != plaintext");
    CHECK(wubu_dmc_decrypt_buf(&c, buf, 16) == 0, "decrypt buf ok");
    CHECK(memcmp(buf, save, 16) == 0, "buffer round-trips exactly");

    /* different key -> different ciphertext */
    uint32_t key2[4] = { 0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0 };
    wubu_dmcrypt_t c2;
    wubu_dmc_init(&c2, key2);
    uint32_t p[2] = { 0xAABBCCDD, 0xEEFF0011 };
    uint32_t p2[2] = { 0xAABBCCDD, 0xEEFF0011 };
    wubu_dmc_encrypt(&c, p);
    wubu_dmc_encrypt(&c2, p2);
    CHECK(p[0] != p2[0] || p[1] != p2[1], "different keys -> different ciphertext");

    /* decrypting with the wrong key does NOT recover plaintext */
    uint32_t wrong[2] = { 0xAABBCCDD, 0xEEFF0011 };
    wubu_dmc_encrypt(&c, wrong);
    wubu_dmc_decrypt(&c2, wrong);   /* wrong key */
    CHECK(wrong[0] != 0xAABBCCDD || wrong[1] != 0xEEFF0011, "wrong key does not decrypt");

    /* bad length (not multiple of 8) */
    CHECK(wubu_dmc_encrypt_buf(&c, buf, 15) == -1, "non-block-multiple length -> -1");

    /* fail-closed on NULL */
    CHECK(wubu_dmc_init(NULL, key) == -1, "NULL init -> -1");
    CHECK(wubu_dmc_encrypt_buf(&c, NULL, 8) == -1, "NULL buf -> -1");

    printf("\n=== test_dmcrypt: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
