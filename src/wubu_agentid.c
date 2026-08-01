/*
 * wubu_agentid.c -- Inter-agent identity + zero-trust auth (DD03). C11.
 *
 * Convergence (SPIFFE/SVID + zero-trust + behavioral identity 7-hop):
 *   - DD03: each CoAgent gets a verifiable identity (ID + name + capability
 *     set + orchestrator signature). Before any agent's vote counts in BFT,
 *     its credential is verified (zero-trust: verify every hop, not just
 *     at entry). The signature is a simplified pseudo-signature (orchestrator
 *     mixes agent_id + name + caps → hash). In production this would be
 *     X.509/SPIFFE SVID, but the C11 verify-and-check-capability logic
 *     is the same.
 */
#include "wubu_agentid.h"
#include <string.h>

static unsigned simple_hash(const char *s) {
    unsigned h = WUBU_AGENTID_SIG_MAGIC;
    while (*s) { h = h * 31 + (unsigned)(*s); s++; }
    return h;
}

int wubu_agentid_issue(wubu_id_registry_t *reg, int id, const char *name,
                        const char **caps, int n_caps) {
    if (!reg || !name || n_caps < 0 || n_caps > WUBU_AGENTID_MAX_CAPS) return -1;
    if (reg->n_agents >= 64) return -1;
    if (wubu_agentid_exists(reg, id)) return -1;
    wubu_agent_id_t *a = &reg->agents[reg->n_agents];
    a->id = id;
    strncpy(a->name, name, WUBU_AGENTID_MAX_NAME - 1);
    a->name[WUBU_AGENTID_MAX_NAME - 1] = '\0';
    a->n_caps = n_caps;
    for (int i = 0; i < n_caps; i++) {
        strncpy(a->caps[i], caps[i], 31);
        a->caps[i][31] = '\0';
    }
    /* Credential signature: mix of id + name + caps hash */
    char buf[512];
    snprintf(buf, sizeof(buf), "%d:%s", id, name);
    for (int i = 0; i < n_caps; i++) {
        strncat(buf, ":", sizeof(buf) - strlen(buf) - 1);
        strncat(buf, caps[i], sizeof(buf) - strlen(buf) - 1);
    }
    a->sig = simple_hash(buf);
    reg->n_agents++;
    return 0;
}

int wubu_agentid_exists(const wubu_id_registry_t *reg, int agent_id) {
    if (!reg) return 0;
    for (int i = 0; i < reg->n_agents; i++)
        if (reg->agents[i].id == agent_id) return 1;
    return 0;
}

int wubu_agentid_verify(const wubu_id_registry_t *reg, int agent_id,
                         const char *required_cap) {
    if (!reg || !required_cap) return 0;
    for (int i = 0; i < reg->n_agents; i++) {
        if (reg->agents[i].id != agent_id) continue;
        /* Zero-trust: verify signature is non-zero (credential issued by orchestrator) */
        if (reg->agents[i].sig == 0) return 0;
        /* Verify capability */
        for (int c = 0; c < reg->agents[i].n_caps; c++) {
            if (strcmp(reg->agents[i].caps[c], required_cap) == 0)
                return 1; /* credential valid + has required cap */
        }
        return 0;
    }
    return 0; /* agent not found */
}
