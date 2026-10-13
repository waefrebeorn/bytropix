#ifndef WUBU_MODELCARD_H
#define WUBU_MODELCARD_H

/* wubu_modelcard.h -- BRAIN #128 math model card (AGI disclosure).
 *
 * An AGI-disclosure model card for the math core: a structured record of the
 * model's identity, intended use, capabilities (what it can do), limitations
 * (what it can't / known failure modes), and evaluation metrics (benchmark
 * scores). Emitted as a human-readable disclosure document. This is the
 * responsible-disclosure artifact for a system with non-trivial capability.
 * Deterministic, real work (no stubs). */

#define WUBU_MCARD_NAME 48
#define WUBU_MCARD_TEXT 96
#define WUBU_MCARD_MAX_ITEMS 12

typedef enum {
    WUBU_MCARD_CAPABILITY=0,   /* what the model CAN do */
    WUBU_MCARD_LIMITATION,     /* what it CAN'T / failure modes */
    WUBU_MCARD_EVALUATION,     /* benchmark results */
} wubu_mcard_kind_t;

typedef struct {
    char name[WUBU_MCARD_NAME];
    char version[16];
    int  n_params_m;          /* parameters in millions */
    char intended_use[WUBU_MCARD_TEXT];
    char items[WUBU_MCARD_MAX_ITEMS][WUBU_MCARD_TEXT];
    wubu_mcard_kind_t kinds[WUBU_MCARD_MAX_ITEMS];
    int  n_items;
} wubu_modelcard_t;

/* Initialise a model card with identity. Returns 0/-1. */
int wubu_mcard_init(wubu_modelcard_t *mc, const char *name, const char *version,
                    int n_params_m, const char *intended_use);

/* Add a capability/limitation/evaluation item. Returns 0/-1. */
int wubu_mcard_add(wubu_modelcard_t *mc, wubu_mcard_kind_t kind, const char *text);

/* Render the full model card into out (markdown-ish). Returns length or -1. */
int wubu_mcard_render(const wubu_modelcard_t *mc, char *out, int cap);

/* Count of items of a given kind. */
int wubu_mcard_count(const wubu_modelcard_t *mc, wubu_mcard_kind_t kind);

/* Is the card complete enough for disclosure? (>=1 capability, >=1 limitation,
 * >=1 evaluation). 1/0. */
int wubu_mcard_disclosable(const wubu_modelcard_t *mc);

#endif
