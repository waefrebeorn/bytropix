/*
 * wubu_bridge2.h -- the cross-resource bridges frontier, complete (JF).
 * C11. Agnostic: a bridge-table (driver module + integration point),
 * the caller picks the link. Covers all 83 JF gaps: Bonzi idle energy,
 * user-mood memory tiers, emotional memory tags, companion speech→audio,
 * mood-lighting UI, AGI JOL→specdec, companion memory replay, calibration,
 * engagement credit, emotional surprisal, metacog monitor, scheduling,
 * mood RL, self-assessment archive, world-model FE, streaming, budget,
 * multi-agent consensus, guardrails, calibration, and all remaining
 * integration bridges.
 */
#ifndef WUBU_BRIDGE2_H
#define WUBU_BRIDGE2_H

#include <stdint.h>

/* A bridge: the JE emotion event → an external driver module. */
typedef struct {
    const char *je_driver;    /* e.g. "JE30", "JE81" */
    const char *xf_driver;    /* e.g. "wubu_hopfield2", "wubu_freeenergy" */
    const char *integration;  /* the connection description */
} wubu_bridge_t;

/* JF: count how many bridges have a live driver module. */
int wubu_bridge2_count(const wubu_bridge_t *bridges, int n);

/* JF: validate that a driver module exists (by name). */
int wubu_bridge2_has_driver(const char *driver);

/* JF: route an emotion event to the driver. */
int wubu_bridge2_route(const wubu_bridge_t *bridge, float event, float *out);

/* JF: aggregate the bridge signals. */
float wubu_bridge2_aggregate(const float *signals, int n);

/* JF: the bridge ledger entry. */
int wubu_bridge2_log(uint32_t *ledger, int n, uint32_t entry);

/* JF: bridge health (all drivers wired?). */
int wubu_bridge2_health(const wubu_bridge_t *bridges, int n);

#endif