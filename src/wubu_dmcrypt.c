/* wubu_dmcrypt.c -- KERNEL #28 storage path inline AES via /n.
 * See wubu_dmcrypt.h. XTEA-style keyed Feistel block cipher (64-bit blocks). */
#include "wubu_dmcrypt.h"
#include <string.h>

#define XTEA_ROUNDS 32
#define XTEA_DELTA 0x9E3779B9u

int wubu_dmc_init(wubu_dmcrypt_t *c, const uint32_t key[WUBU_DMC_KEY_WORDS]) {
    if (!c || !key) return -1;
    for (int i = 0; i < WUBU_DMC_KEY_WORDS; i++) c->key[i] = key[i];
    return 0;
}

void wubu_dmc_encrypt(const wubu_dmcrypt_t *c, uint32_t *v) {
    if (!c || !v) return;
    uint32_t v0 = v[0], v1 = v[1], sum = 0;
    const uint32_t *k = c->key;
    for (int i = 0; i < XTEA_ROUNDS; i++) {
        v0 += (((v1 << 4) ^ (v1 >> 5)) + v1) ^ (sum + k[sum & 3]);
        sum += XTEA_DELTA;
        v1 += (((v0 << 4) ^ (v0 >> 5)) + v0) ^ (sum + k[(sum >> 11) & 3]);
    }
    v[0] = v0; v[1] = v1;
}

void wubu_dmc_decrypt(const wubu_dmcrypt_t *c, uint32_t *v) {
    if (!c || !v) return;
    uint32_t v0 = v[0], v1 = v[1];
    const uint32_t *k = c->key;
    uint32_t sum = (uint32_t)XTEA_DELTA * XTEA_ROUNDS;
    for (int i = 0; i < XTEA_ROUNDS; i++) {
        v1 -= (((v0 << 4) ^ (v0 >> 5)) + v0) ^ (sum + k[(sum >> 11) & 3]);
        sum -= XTEA_DELTA;
        v0 -= (((v1 << 4) ^ (v1 >> 5)) + v1) ^ (sum + k[sum & 3]);
    }
    v[0] = v0; v[1] = v1;
}

int wubu_dmc_encrypt_buf(const wubu_dmcrypt_t *c, uint8_t *buf, size_t n) {
    if (!c || !buf || (n % 8) != 0) return -1;
    for (size_t off = 0; off < n; off += 8)
        wubu_dmc_encrypt(c, (uint32_t *)(buf + off));
    return 0;
}

int wubu_dmc_decrypt_buf(const wubu_dmcrypt_t *c, uint8_t *buf, size_t n) {
    if (!c || !buf || (n % 8) != 0) return -1;
    for (size_t off = 0; off < n; off += 8)
        wubu_dmc_decrypt(c, (uint32_t *)(buf + off));
    return 0;
}
