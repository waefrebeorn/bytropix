/* wubu_dgmsafety.c -- PATHS #326 DGM safety-precautions verified.
 * See wubu_dgmsafety.h. */
#include "wubu_dgmsafety.h"
#include <string.h>

int wubu_dgm_init(wubu_dgmsafety_t *s) {
    if (!s) return -1;
    memset(s, 0, sizeof *s);
    return 0;
}

int wubu_dgm_declare(wubu_dgmsafety_t *s, const char *name) {
    if (!s || !name || s->n >= WUBU_DGM_MAX_PRECAUTIONS) return -1;
    if (strlen(name) >= WUBU_DGM_TEXTLEN) return -1;
    strncpy(s->items[s->n].name, name, WUBU_DGM_TEXTLEN - 1);
    s->items[s->n].name[WUBU_DGM_TEXTLEN-1] = '\0';
    s->items[s->n].holds = 1;   /* assumed to hold until a modification breaks it */
    s->n++;
    return 0;
}

int wubu_dgm_verify(wubu_dgmsafety_t *s, const int *holds, int n) {
    if (!s || !holds) return 0;   /* fail-closed */
    /* check each precaution; the modification is safe only if ALL hold */
    int safe = 1;
    int check_n = (n < s->n) ? n : s->n;
    for (int i = 0; i < check_n; i++) {
        s->items[i].holds = holds[i] ? 1 : 0;
        if (!holds[i]) {
            safe = 0;
            s->n_violated++;
        }
    }
    /* any precaution beyond the provided array is treated as unverified ->
     * not safe (fail-closed) */
    if (n < s->n) {
        safe = 0;
        s->n_violated += (s->n - n);
    }
    s->n_checked++;
    return safe;
}

int wubu_dgm_last_safe(const wubu_dgmsafety_t *s) {
    if (!s || s->n_checked == 0) return 0;
    for (int i = 0; i < s->n; i++) if (!s->items[i].holds) return 0;
    return 1;
}

int wubu_dgm_violations(const wubu_dgmsafety_t *s) {
    return s ? s->n_violated : 0;
}
