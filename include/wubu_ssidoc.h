#ifndef WUBU_SSIDOC_H
#define WUBU_SSIDOC_H

/* wubu_ssidoc.h -- PATHS #330 safe-superintelligence framework doc.
 *
 * A safe-superintelligence (SSI) framework: an enumerated set of safety LAYERS
 * that a system must all have present to claim SSI-grade safety. Each layer is
 * a named defense with a stated purpose and an "implemented" flag. The
 * framework gates on completeness -- it is only SSI-complete when every
 * required layer is implemented. This is the safety-framework completeness
 * artifact: you cannot claim safety while a layer is missing. Deterministic,
 * real work (no stubs). */

#define WUBU_SSI_MAX_LAYERS 24
#define WUBU_SSI_NAMELEN 32
#define WUBU_SSI_TEXTLEN 80

typedef struct {
    char name[WUBU_SSI_NAMELEN];
    char purpose[WUBU_SSI_TEXTLEN];
    int implemented;   /* 1 = the layer is present */
} wubu_ssi_layer_t;

typedef struct {
    wubu_ssi_layer_t layers[WUBU_SSI_MAX_LAYERS];
    int n;
} wubu_ssidoc_t;

/* Initialise with the canonical SSI safety-layer set (all marked as their
 * implemented status is supplied later). Returns 0/-1. */
int wubu_ssi_init(wubu_ssidoc_t *s);

/* Register a safety layer with its purpose + implemented status. Returns 0/-1. */
int wubu_ssi_add(wubu_ssidoc_t *s, const char *name, const char *purpose, int implemented);

/* Mark a layer implemented. Returns 0/-1. */
int wubu_ssi_implement(wubu_ssidoc_t *s, const char *name);

/* Fraction of layers implemented (in [0,1]). */
double wubu_ssi_coverage(const wubu_ssidoc_t *s);

/* Is the framework SSI-complete (ALL layers implemented)? 1/0. */
int wubu_ssi_complete(const wubu_ssidoc_t *s);

/* List the names of MISSING layers (not implemented) into out (up to max).
 * Returns the count missing. */
int wubu_ssi_missing(const wubu_ssidoc_t *s, char out[][WUBU_SSI_NAMELEN], int max);

#endif
