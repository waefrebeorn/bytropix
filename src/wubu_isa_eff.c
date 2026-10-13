/* wubu_isa_eff.c -- PATHS #323 custom-ISA compiler efficiency path.
 * See wubu_isa_eff.h. */
#include "wubu_isa_eff.h"
#include <string.h>

int wubu_isa_eff_compare(wubu_isa_eff_t *e, const char *name,
                         int custom_instrs, double custom_cpi,
                         int portable_instrs, double portable_cpi, double portable_overhead) {
    if (!e || !name || custom_instrs < 0 || portable_instrs < 0) return -1;
    if (custom_cpi <= 0 || portable_cpi <= 0) return -1;
    if (strlen(name) >= 40) return -1;
    strncpy(e->custom.name, name, 39); e->custom.name[39] = '\0';
    strncpy(e->portable.name, name, 39); e->portable.name[39] = '\0';
    e->custom.n_instrs = custom_instrs;
    e->custom.cycles_per_instr = custom_cpi;
    e->custom.abi_overhead = 0.0;   /* custom ISA has no ABI translation */
    e->portable.n_instrs = portable_instrs;
    e->portable.cycles_per_instr = portable_cpi;
    e->portable.abi_overhead = portable_overhead >= 0 ? portable_overhead : 0.0;
    return 0;
}

double wubu_isa_eff_cycles(const wubu_isa_compile_t *c) {
    if (!c) return 0.0;
    return c->n_instrs * c->cycles_per_instr + c->abi_overhead;
}

double wubu_isa_eff_speedup(const wubu_isa_eff_t *e) {
    if (!e) return 0.0;
    double custom = wubu_isa_eff_cycles(&e->custom);
    double portable = wubu_isa_eff_cycles(&e->portable);
    if (custom <= 0.0) return 0.0;
    return portable / custom;
}

int wubu_isa_eff_custom_wins(const wubu_isa_eff_t *e) {
    if (!e) return 0;
    return wubu_isa_eff_cycles(&e->custom) < wubu_isa_eff_cycles(&e->portable) ? 1 : 0;
}
