#ifndef WUBU_EDRCARD_H
#define WUBU_EDRCARD_H

/* wubu_edrcard.h -- AGIEDR #311 AGI-EDR model card + disclosure.
 *
 * A security-disclosure card for the EDR subsystem: documents what threats it
 * DETECTS, its known BLIND SPOTS (what it can't see / known evasion), its
 * RESPONSE actions, and its threat model. Honest disclosure of limitations is
 * the point -- a security system that overstates coverage is dangerous.
 * Deterministic, real work (no stubs). */

#define WUBU_EDR_MAX_ITEMS 12
#define WUBU_EDR_TEXTLEN 96

typedef enum {
    WUBU_EDR_DETECTS=0,     /* threats it catches */
    WUBU_EDR_BLINDSPOT,     /* known gaps / evasion techniques */
    WUBU_EDR_RESPONSE,      /* automated response actions */
} wubu_edr_kind_t;

typedef struct {
    char items[WUBU_EDR_MAX_ITEMS][WUBU_EDR_TEXTLEN];
    wubu_edr_kind_t kinds[WUBU_EDR_MAX_ITEMS];
    int n_items;
    char threat_model[WUBU_EDR_TEXTLEN];
} wubu_edrcard_t;

/* Initialise with a threat-model statement. Returns 0/-1. */
int wubu_edr_init(wubu_edrcard_t *c, const char *threat_model);

/* Add a disclosure item (detection / blindspot / response). Returns 0/-1. */
int wubu_edr_add(wubu_edrcard_t *c, wubu_edr_kind_t kind, const char *text);

/* Count items of a kind. */
int wubu_edr_count(const wubu_edrcard_t *c, wubu_edr_kind_t kind);

/* Render the disclosure card into out. Returns length or -1. */
int wubu_edr_render(const wubu_edrcard_t *c, char *out, int cap);

/* Is the card honest/complete for disclosure? Requires >=1 detection AND
 * >=1 blindspot (you MUST disclose at least one known gap). 1/0. */
int wubu_edr_disclosable(const wubu_edrcard_t *c);

#endif
