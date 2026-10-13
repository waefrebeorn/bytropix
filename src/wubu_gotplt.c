/* wubu_gotplt.c -- RUNTIME #346 dynamic linker GOT/PLT relocation.
 * See wubu_gotplt.h. */
#include "wubu_gotplt.h"
#include <string.h>

int wubu_gp_init(wubu_gotplt_t *g, wubu_gp_bind_t bind, unsigned long long resolver_addr) {
    if (!g) return -1;
    memset(g->syms, 0, sizeof g->syms);
    memset(g->relocs, 0, sizeof g->relocs);
    g->n_syms = 0;
    g->n_relocs = 0;
    g->bind = bind;
    g->resolver_addr = resolver_addr;
    return 0;
}

int wubu_gp_add_sym(wubu_gotplt_t *g, const char *name, unsigned long long addr) {
    if (!g || !name || g->n_syms >= WUBU_GP_MAX) return -1;
    if (strlen(name) >= 32) return -1;
    strncpy(g->syms[g->n_syms].name, name, 31); g->syms[g->n_syms].name[31] = '\0';
    g->syms[g->n_syms].addr = addr;
    g->n_syms++;
    return 0;
}

int wubu_gp_add_reloc(wubu_gotplt_t *g, const char *name, int is_func) {
    if (!g || !name || g->n_relocs >= WUBU_GP_MAX) return -1;
    if (strlen(name) >= 32) return -1;
    strncpy(g->relocs[g->n_relocs].name, name, 31); g->relocs[g->n_relocs].name[31] = '\0';
    g->relocs[g->n_relocs].is_func = is_func ? 1 : 0;
    g->relocs[g->n_relocs].got = 0;
    g->relocs[g->n_relocs].plt = 0;
    g->relocs[g->n_relocs].bound = 0;
    g->n_relocs++;
    return 0;
}

static const wubu_gp_sym_t *lookup_sym(const wubu_gotplt_t *g, const char *name) {
    for (int i = 0; i < g->n_syms; i++)
        if (strcmp(g->syms[i].name, name) == 0) return &g->syms[i];
    return NULL;
}

static wubu_gp_reloc_t *lookup_reloc(wubu_gotplt_t *g, const char *name) {
    for (int i = 0; i < g->n_relocs; i++)
        if (strcmp(g->relocs[i].name, name) == 0) return &g->relocs[i];
    return NULL;
}

int wubu_gp_link(wubu_gotplt_t *g) {
    if (!g) return -1;
    int resolved = 0;
    for (int i = 0; i < g->n_relocs; i++) {
        wubu_gp_reloc_t *r = &g->relocs[i];
        const wubu_gp_sym_t *s = lookup_sym(g, r->name);
        if (!s) continue;   /* undefined symbol: leave unbound (would be weak/error) */
        if (g->bind == WUBU_GP_NOW) {
            /* eager: GOT holds the real address; PLT jumps straight there */
            r->got = s->addr;
            r->plt = r->is_func ? s->addr : 0;
            r->bound = 1;
            resolved++;
        } else {
            /* LAZY: data relocs are resolved now; function relocs point at the
             * resolver until first call */
            if (!r->is_func) {
                r->got = s->addr;
                r->plt = 0;
                r->bound = 1;
                resolved++;
            } else {
                r->got = g->resolver_addr;   /* GOT -> resolver (lazy) */
                r->plt = g->resolver_addr;
                r->bound = 0;
            }
        }
    }
    return resolved;
}

int wubu_gp_bind(wubu_gotplt_t *g, const char *name) {
    if (!g || !name) return -1;
    wubu_gp_reloc_t *r = lookup_reloc(g, name);
    if (!r || !r->is_func) return -1;
    const wubu_gp_sym_t *s = lookup_sym(g, name);
    if (!s) return -1;
    r->got = s->addr;   /* resolve the GOT entry */
    r->plt = s->addr;
    r->bound = 1;
    return 0;
}

int wubu_gp_is_bound(const wubu_gotplt_t *g, const char *name) {
    if (!g || !name) return 0;
    for (int i = 0; i < g->n_relocs; i++)
        if (strcmp(g->relocs[i].name, name) == 0) return g->relocs[i].bound;
    return 0;
}
