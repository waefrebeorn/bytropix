/* wubu_bonzi_mem.c -- UX #239 companion persistent memory.
 * See wubu_bonzi_mem.h. */
#include "wubu_bonzi_mem.h"
#include <string.h>
#include <stdint.h>

int wubu_bonzi_mem_init(wubu_bonzi_mem_t *m) {
    if (!m) return -1;
    memset(m->entries, 0, sizeof m->entries);
    m->n = 0;
    m->next_seq = 1;
    return 0;
}

static int find(const wubu_bonzi_mem_t *m, const char *key) {
    for (int i = 0; i < m->n; i++)
        if (strcmp(m->entries[i].key, key) == 0) return i;
    return -1;
}

int wubu_bonzi_mem_set(wubu_bonzi_mem_t *m, const char *key, const char *val) {
    if (!m || !key || !val) return -1;
    if (strlen(key) >= WUBU_BONZI_KEYLEN || strlen(val) >= WUBU_BONZI_VALLEN) return -1;
    int i = find(m, key);
    if (i >= 0) {
        strncpy(m->entries[i].val, val, WUBU_BONZI_VALLEN - 1);
        m->entries[i].val[WUBU_BONZI_VALLEN - 1] = '\0';
        m->entries[i].seq = m->next_seq++;   /* refresh recency */
        return 0;
    }
    if (m->n >= WUBU_BONZI_MEM_MAX) {
        /* evict oldest (lowest seq) */
        int oldest = 0;
        for (int j = 1; j < m->n; j++)
            if (m->entries[j].seq < m->entries[oldest].seq) oldest = j;
        i = oldest;
    } else {
        i = m->n++;
    }
    strncpy(m->entries[i].key, key, WUBU_BONZI_KEYLEN - 1);
    m->entries[i].key[WUBU_BONZI_KEYLEN - 1] = '\0';
    strncpy(m->entries[i].val, val, WUBU_BONZI_VALLEN - 1);
    m->entries[i].val[WUBU_BONZI_VALLEN - 1] = '\0';
    m->entries[i].seq = m->next_seq++;
    return 0;
}

const char *wubu_bonzi_mem_get(const wubu_bonzi_mem_t *m, const char *key) {
    if (!m || !key) return NULL;
    int i = find(m, key);
    return i >= 0 ? m->entries[i].val : NULL;
}

int wubu_bonzi_mem_forget(wubu_bonzi_mem_t *m, const char *key) {
    if (!m || !key) return -1;
    int i = find(m, key);
    if (i < 0) return -1;
    for (int j = i; j < m->n - 1; j++) m->entries[j] = m->entries[j + 1];
    m->n--;
    return 0;
}

int wubu_bonzi_mem_count(const wubu_bonzi_mem_t *m) {
    return m ? m->n : 0;
}

int wubu_bonzi_mem_serialize(const wubu_bonzi_mem_t *m, unsigned char *out, size_t outsz) {
    if (!m || !out) return -1;
    /* format: [n:u32][next_seq:u64] then per entry [seq:u64][klen:u16][key][vlen:u16][val] */
    size_t need = 4 + 8;
    for (int i = 0; i < m->n; i++)
        need += 8 + 2 + strlen(m->entries[i].key) + 2 + strlen(m->entries[i].val);
    if (need > outsz) return -1;
    unsigned char *p = out;
    uint32_t n = (uint32_t)m->n;
    memcpy(p, &n, 4); p += 4;
    memcpy(p, &m->next_seq, 8); p += 8;
    for (int i = 0; i < m->n; i++) {
        memcpy(p, &m->entries[i].seq, 8); p += 8;
        uint16_t kl = (uint16_t)strlen(m->entries[i].key);
        memcpy(p, &kl, 2); p += 2;
        memcpy(p, m->entries[i].key, kl); p += kl;
        uint16_t vl = (uint16_t)strlen(m->entries[i].val);
        memcpy(p, &vl, 2); p += 2;
        memcpy(p, m->entries[i].val, vl); p += vl;
    }
    return (int)(p - out);
}

int wubu_bonzi_mem_restore(wubu_bonzi_mem_t *m, const unsigned char *buf, size_t bufsz) {
    if (!m || !buf) return -1;
    if (bufsz < 12) return -1;
    const unsigned char *p = buf;
    uint32_t n; memcpy(&n, p, 4); p += 4;
    if (n > WUBU_BONZI_MEM_MAX) return -1;
    uint64_t next_seq; memcpy(&next_seq, p, 8); p += 8;
    wubu_bonzi_mem_t tmp;
    wubu_bonzi_mem_init(&tmp);
    for (uint32_t i = 0; i < n; i++) {
        if ((size_t)(p - buf) + 10 > bufsz) return -1;
        uint64_t seq; memcpy(&seq, p, 8); p += 8;
        uint16_t kl; memcpy(&kl, p, 2); p += 2;
        if (kl >= WUBU_BONZI_KEYLEN || (size_t)(p - buf) + kl + 2 > bufsz) return -1;
        char key[WUBU_BONZI_KEYLEN]; memcpy(key, p, kl); key[kl] = '\0'; p += kl;
        uint16_t vl; memcpy(&vl, p, 2); p += 2;
        if (vl >= WUBU_BONZI_VALLEN || (size_t)(p - buf) + vl > bufsz) return -1;
        char val[WUBU_BONZI_VALLEN]; memcpy(val, p, vl); val[vl] = '\0'; p += vl;
        if (wubu_bonzi_mem_set(&tmp, key, val) != 0) return -1;
        tmp.entries[tmp.n - 1].seq = seq;
    }
    tmp.next_seq = next_seq;
    *m = tmp;
    return 0;
}
