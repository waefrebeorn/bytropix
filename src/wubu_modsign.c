/* wubu_modsign.c -- KERNEL #23 kernel modules signed + verified on load.
 * See wubu_modsign.h. */
#include "wubu_modsign.h"
#include <string.h>

int wubu_ms_init(wubu_modsign_t *s, unsigned long long kernel_key) {
    if (!s) return -1;
    memset(s->mods, 0, sizeof s->mods);
    s->n = 0;
    s->kernel_key = kernel_key;
    return 0;
}

int wubu_ms_enroll(wubu_modsign_t *s, const char *name, unsigned long long signature) {
    if (!s || !name || s->n >= WUBU_MS_MAX) return -1;
    if (strlen(name) >= 32) return -1;
    wubu_ms_mod_t *m = &s->mods[s->n];
    strncpy(m->name, name, 31); m->name[31] = '\0';
    m->signature = signature;
    /* the trusted signature is the enrolled signature combined with the key */
    m->trusted = signature ^ s->kernel_key;
    m->loaded = 0;
    s->n++;
    return 0;
}

static wubu_ms_mod_t *find(wubu_modsign_t *s, const char *name) {
    for (int i = 0; i < s->n; i++)
        if (strcmp(s->mods[i].name, name) == 0) return &s->mods[i];
    return NULL;
}

int wubu_ms_load(wubu_modsign_t *s, const char *name, unsigned long long signature) {
    if (!s || !name) return -1;
    wubu_ms_mod_t *m = find(s, name);
    if (!m) return 0;   /* unknown module: not enrolled -> reject */
    /* verify: the presented signature, combined with the key, must match the
     * enrolled trusted signature */
    if ((signature ^ s->kernel_key) != m->trusted) return 0;   /* mismatch -> reject */
    m->loaded = 1;
    return 1;
}

int wubu_ms_is_loaded(const wubu_modsign_t *s, const char *name) {
    if (!s || !name) return 0;
    for (int i = 0; i < s->n; i++)
        if (strcmp(s->mods[i].name, name) == 0) return s->mods[i].loaded;
    return 0;
}
