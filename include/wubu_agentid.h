/*
 * wubu_agentid.h -- Inter-agent identity + zero-trust auth (DD03).
 */
#ifndef WUBU_AGENTID_H
#define WUBU_AGENTID_H

#define WUBU_AGENTID_MAX_CAPS 16
#define WUBU_AGENTID_MAX_NAME 64
#define WUBU_AGENTID_SIG_MAGIC 0xA7B4C9D1U

typedef struct {
    int   id;
    char  name[WUBU_AGENTID_MAX_NAME];
    char  caps[WUBU_AGENTID_MAX_CAPS][32];  /* capabilities: "read_kv", "spawn", etc */
    int   n_caps;
    unsigned sig;  /* "credential signature" — signed by orchestrator */
} wubu_agent_id_t;

typedef struct {
    wubu_agent_id_t agents[64];
    int n_agents;
} wubu_id_registry_t;

/* Issue a verified identity: orchestrator signs the agent's claim. */
int wubu_agentid_issue(wubu_id_registry_t *reg, int id, const char *name,
                        const char **caps, int n_caps);
/* Verify an agent's credential before accepting its vote (zero-trust). */
int wubu_agentid_verify(const wubu_id_registry_t *reg, int agent_id,
                         const char *required_cap);
/* Check if registry has a given agent. */
int wubu_agentid_exists(const wubu_id_registry_t *reg, int agent_id);

#endif