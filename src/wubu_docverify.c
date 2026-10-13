/* wubu_docverify.c -- STACK #369 docs-vs-code verification.
 * See wubu_docverify.h. */
#include "wubu_docverify.h"
#include <string.h>

int wubu_dv_init(wubu_docverify_t *v) {
    if (!v) return -1;
    memset(v->mods, 0, sizeof v->mods);
    v->n = 0;
    return 0;
}

int wubu_dv_record(wubu_docverify_t *v, const char *name,
                   int in_code, int in_header, int in_docs) {
    if (!v || !name || v->n >= WUBU_DV_MAX) return -1;
    if (strlen(name) >= 64) return -1;
    wubu_dv_module_t *m = &v->mods[v->n];
    strncpy(m->name, name, 63); m->name[63] = '\0';
    m->in_code = in_code ? 1 : 0;
    m->in_header = in_header ? 1 : 0;
    m->in_docs = in_docs ? 1 : 0;
    v->n++;
    return 0;
}

int wubu_dv_drift(const wubu_docverify_t *v) {
    if (!v) return 0;
    int drift = 0;
    for (int i = 0; i < v->n; i++) {
        /* documented but not backed by code = drift */
        if (v->mods[i].in_docs && !v->mods[i].in_code) drift++;
    }
    return drift;
}

int wubu_dv_backed(const wubu_docverify_t *v) {
    if (!v) return 0;
    int backed = 0;
    for (int i = 0; i < v->n; i++) {
        if (v->mods[i].in_docs && v->mods[i].in_code && v->mods[i].in_header)
            backed++;
    }
    return backed;
}

double wubu_dv_match_rate(const wubu_docverify_t *v) {
    if (!v) return 0.0;
    int documented = 0;
    for (int i = 0; i < v->n; i++) if (v->mods[i].in_docs) documented++;
    if (documented == 0) return 1.0;   /* nothing documented -> vacuously consistent */
    return (double)wubu_dv_backed(v) / (double)documented;
}

int wubu_dv_consistent(const wubu_docverify_t *v) {
    if (!v) return 0;
    return wubu_dv_drift(v) == 0 ? 1 : 0;
}
