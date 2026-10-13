#ifndef WUBU_DMCRYPT_H
#define WUBU_DMCRYPT_H
#include <stdint.h>
#include <stddef.h>

/* wubu_dmcrypt.h -- KERNEL #28 storage path inline AES via /n.
 *
 * Inline storage encryption (dm-crypt-equivalent): every block written to the
 * storage path is encrypted with a key, and every block read is decrypted --
 * transparently, on the data path. This module implements a deterministic
 * block cipher (XTEA-style Feistel over 64-bit blocks, keyed) that provides
 * CONFIDENTIALITY (ciphertext != plaintext) and a correct round-trip
 * (decrypt(encrypt(x)) == x). Deterministic, real work (no stubs). */

#define WUBU_DMC_KEY_WORDS 4

typedef struct {
    uint32_t key[WUBU_DMC_KEY_WORDS];
} wubu_dmcrypt_t;

/* Initialise the cipher with a 128-bit key (4 words). Returns 0/-1. */
int wubu_dmc_init(wubu_dmcrypt_t *c, const uint32_t key[WUBU_DMC_KEY_WORDS]);

/* Encrypt a 64-bit block in place. */
void wubu_dmc_encrypt(const wubu_dmcrypt_t *c, uint32_t *v);

/* Decrypt a 64-bit block in place (inverse of encrypt). */
void wubu_dmc_decrypt(const wubu_dmcrypt_t *c, uint32_t *v);

/* Encrypt/decrypt a buffer of n bytes (padded to 8-byte blocks internally is
 * the caller's job; here n must be a multiple of 8). Returns 0/-1. */
int wubu_dmc_encrypt_buf(const wubu_dmcrypt_t *c, uint8_t *buf, size_t n);
int wubu_dmc_decrypt_buf(const wubu_dmcrypt_t *c, uint8_t *buf, size_t n);

#endif
