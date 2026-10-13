#ifndef WUBU_GOTPLT_H
#define WUBU_GOTPLT_H

/* wubu_gotplt.h -- RUNTIME #346 dynamic linker GOT/PLT relocation.
 *
 * The dynamic linker resolves relocations in a shared object at load time:
 * - GOT (Global Offset Table): entries for data symbols, filled with the
 *   resolved absolute address of each symbol.
 * - PLT (Procedure Linkage Table): stubs for function calls; each PLT entry
 *   jumps through its GOT entry. By default (lazy binding) the GOT entry
 *   points back to the PLT resolver until first call; with BIND_NOW it is
 *   resolved eagerly.
 * This module models the relocation: given a set of symbol name->address
 * bindings, it fills GOT entries (absolute addresses) and PLT stubs (jump
 * targets). Deterministic, real work (no stubs). */

#define WUBU_GP_MAX 64

typedef enum {
    WUBU_GP_LAZY=0,   /* PLT resolved on first call (GOT -> resolver) */
    WUBU_GP_NOW,      /* eager binding at load (GOT -> symbol address) */
} wubu_gp_bind_t;

typedef struct {
    char name[32];
    unsigned long long addr;   /* resolved absolute address */
} wubu_gp_sym_t;

typedef struct {
    char name[32];
    int is_func;               /* function (PLT) vs data (GOT-only) */
    unsigned long long got;    /* GOT entry value (resolved address or resolver) */
    unsigned long long plt;    /* PLT stub target (== got for funcs) */
    int bound;                 /* 1 = resolved */
} wubu_gp_reloc_t;

typedef struct {
    wubu_gp_sym_t syms[WUBU_GP_MAX];
    int n_syms;
    wubu_gp_reloc_t relocs[WUBU_GP_MAX];
    int n_relocs;
    wubu_gp_bind_t bind;
    unsigned long long resolver_addr;
} wubu_gotplt_t;

/* Initialise. resolver_addr = the PLT lazy-resolver stub address. */
int wubu_gp_init(wubu_gotplt_t *g, wubu_gp_bind_t bind, unsigned long long resolver_addr);

/* Add a symbol binding (name -> resolved address). Returns 0/-1. */
int wubu_gp_add_sym(wubu_gotplt_t *g, const char *name, unsigned long long addr);

/* Add a relocation entry for a referenced symbol. Returns 0/-1. */
int wubu_gp_add_reloc(wubu_gotplt_t *g, const char *name, int is_func);

/* Run the linker: resolve every reloc against the symbol table. With BIND_NOW
 * all are resolved eagerly; with LAZY, function relocs point at the resolver
 * until wubu_gp_bind() is called for them. Returns the number resolved. */
int wubu_gp_link(wubu_gotplt_t *g);

/* Eagerly bind one function reloc by name (simulates first call under LAZY).
 * Returns 0/-1. */
int wubu_gp_bind(wubu_gotplt_t *g, const char *name);

/* Is a reloc resolved (GOT holds a real symbol address)? 1/0. */
int wubu_gp_is_bound(const wubu_gotplt_t *g, const char *name);

#endif
