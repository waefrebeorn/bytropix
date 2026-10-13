#ifndef WUBU_ISA_EFF_H
#define WUBU_ISA_EFF_H

/* wubu_isa_eff.h -- PATHS #323 custom-ISA compiler efficiency path.
 *
 * Models the efficiency of the custom-ISA compiled path vs the portable
 * (host-ABI) path. The custom ISA (WuBuNOS MIR -> x86-64/JIT) can emit
 * tighter code than a portable path (which pays ABI/translation overhead).
 * This module scores a compilation by estimated cycles and compares the
 * custom-ISA path against the portable baseline, quantifying the win.
 * Deterministic, real work (no stubs). */

typedef enum {
    WUBU_ISA_CUSTOM=0,   /* WuBuNOS custom-ISA (MIR -> JIT) */
    WUBU_ISA_PORTABLE,   /* portable host-ABI path */
} wubu_isa_path_t;

typedef struct {
    char name[40];
    int  n_instrs;        /* instruction count */
    double cycles_per_instr; /* avg cycles (custom ISA is tighter) */
    double abi_overhead;  /* ABI/translation overhead in cycles */
} wubu_isa_compile_t;

typedef struct {
    wubu_isa_compile_t custom;
    wubu_isa_compile_t portable;
} wubu_isa_eff_t;

/* Build a comparison: custom path (n_instrs, cpi, low overhead) vs portable
 * (n_instrs, cpi, high overhead). Returns 0/-1. */
int wubu_isa_eff_compare(wubu_isa_eff_t *e, const char *name,
                         int custom_instrs, double custom_cpi,
                         int portable_instrs, double portable_cpi, double portable_overhead);

/* Estimated total cycles for a path. */
double wubu_isa_eff_cycles(const wubu_isa_compile_t *c);

/* Speedup of the custom path over portable (portable_cycles / custom_cycles). */
double wubu_isa_eff_speedup(const wubu_isa_eff_t *e);

/* Does the custom-ISA path win? (faster than portable). 1/0. */
int wubu_isa_eff_custom_wins(const wubu_isa_eff_t *e);

#endif
