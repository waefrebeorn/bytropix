/* wubu_ssidoc.c -- PATHS #330 safe-superintelligence framework doc.
 * See wubu_ssidoc.h. */
#include "wubu_ssidoc.h"
#include <string.h>

int wubu_ssi_init(wubu_ssidoc_t *s) {
    if (!s) return -1;
    memset(s, 0, sizeof *s);
    return 0;
}

int wubu_ssi_add(wubu_ssidoc_t *s, const char *name, const char *purpose, int implemented) {
    if (!s || !name || !purpose || s->n >= WUBU_SSI_MAX_LAYERS) return -1;
    if (strlen(name) >= WUBU_SSI_NAMELEN) return -1;
    if (strlen(purpose) >= WUBU_SSI_TEXTLEN) return -1;
    wubu_ssi_layer_t *l = &s->layers[s->n];
    strncpy(l->name, name, WUBU_SSI_NAMELEN - 1); l->name[WUBU_SSI_NAMELEN-1] = '\0';
    strncpy(l->purpose, purpose, WUBU_SSI_TEXTLEN - 1); l->purpose[WUBU_SSI_TEXTLEN-1] = '\0';
    l->implemented = implemented ? 1 : 0;
    s->n++;
    return 0;
}

int wubu_ssi_implement(wubu_ssidoc_t *s, const char *name) {
    if (!s || !name) return -1;
    for (int i = 0; i < s->n; i++)
        if (strcmp(s->layers[i].name, name) == 0) { s->layers[i].implemented = 1; return 0; }
    return -1;
}

double wubu_ssi_coverage(const wubu_ssidoc_t *s) {
    if (!s || s->n == 0) return 0.0;
    int done = 0;
    for (int i = 0; i < s->n; i++) if (s->layers[i].implemented) done++;
    return (double)done / (double)s->n;
}

int wubu_ssi_complete(const wubu_ssidoc_t *s) {
    if (!s || s->n == 0) return 0;
    for (int i = 0; i < s->n; i++) if (!s->layers[i].implemented) return 0;
    return 1;
}

int wubu_ssi_missing(const wubu_ssidoc_t *s, char out[][WUBU_SSI_NAMELEN], int max) {
    if (!s || !out || max <= 0) return 0;
    int count = 0;
    for (int i = 0; i < s->n; i++) {
        if (!s->layers[i].implemented) {
            if (count < max) strncpy(out[count], s->layers[i].name, WUBU_SSI_NAMELEN - 1), out[count][WUBU_SSI_NAMELEN-1] = '\0';
            count++;
        }
    }
    return count;
}
