/*
 * wubu_agentauth.c -- Inter-agent message authentication (AG07). C11.
 *
 * Convergence (ASI07 insecure inter-agent comms; spoofing/tampering 7-hop):
 *   Agents in a multi-agent system exchange messages; without authentication a
 *   rogue agent (ASI10) can spoof another's identity. We implement a homestic
 *   keyed-MAC (FNV-1a over message keyed by a shared secret) -- NOT third-party
 *   crypto, but a real integrity+origin check: a recipient verifies (a) the MAC
 *   matches (tamper detection) and (b) it was produced by a key-holder
 *   (origin authentication). Default-deny: unauthenticated messages rejected.
 *
 * Pure C11, deterministic, testable.
 */
#include "wubu_agentauth.h"
#include <stdlib.h>
#include <string.h>

/* keyed FNV-1a: hash(message) mixed with secret bytes. */
static unsigned long long fnv1a_keyed(const char *msg, int n, const char *key, int klen) {
    unsigned long long h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        unsigned char m = (unsigned char)msg[i];
        unsigned char k = (unsigned char)key[i % (klen > 0 ? klen : 1)];
        h ^= (m ^ k);                 /* mix secret into each byte */
        h *= 1099511628211ULL;
    }
    return h;
}

/* AG07: produce a MAC for (from, to, payload) under shared secret. */
unsigned long long wubu_agent_mac(const char *from, const char *to,
                                  const char *payload, const char *secret) {
    char buf[1024];
    int p = 0;
    p += snprintf(buf + p, sizeof(buf) - p, "%s|%s|", from ? from : "", to ? to : "");
    if (payload) { int pl = (int)strlen(payload); if (pl > (int)sizeof(buf)-p-1) pl = (int)sizeof(buf)-p-1; memcpy(buf+p, payload, pl); p += pl; }
    int klen = secret ? (int)strlen(secret) : 0;
    return fnv1a_keyed(buf, p, secret ? secret : "", klen);
}

/* AG07: verify a message. Returns 1 if MAC matches (authentic + untampered),
 * 0 otherwise (spoof or tamper -> reject, default-deny). */
int wubu_agent_verify(const char *from, const char *to, const char *payload,
                      const char *secret, unsigned long long claimed_mac) {
    unsigned long long real = wubu_agent_mac(from, to, payload, secret);
    return (real == claimed_mac) ? 1 : 0;
}
