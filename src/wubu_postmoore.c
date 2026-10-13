/* wubu_postmoore.c -- PATHS #328 post-Moore algorithmic-scaling research.
 * See wubu_postmoore.h. */
#include "wubu_postmoore.h"
#include <math.h>

int wubu_pm_init(wubu_postmoore_t *p, double custom_isa, double spec_decode,
                 double moe_routing, double mem_consolidation) {
    if (!p) return -1;
    if (custom_isa < 1.0 || spec_decode < 1.0 || moe_routing < 1.0 || mem_consolidation < 1.0)
        return -1;   /* each technique must be at least neutral (>=1x) */
    p->custom_isa = custom_isa;
    p->spec_decode = spec_decode;
    p->moe_routing = moe_routing;
    p->mem_consolidation = mem_consolidation;
    return 0;
}

double wubu_pm_total_speedup(const wubu_postmoore_t *p) {
    if (!p) return 0.0;
    return p->custom_isa * p->spec_decode * p->moe_routing * p->mem_consolidation;
}

double wubu_pm_compute_saved(const wubu_postmoore_t *p) {
    if (!p) return 0.0;
    double speedup = wubu_pm_total_speedup(p);
    if (speedup <= 0) return 0.0;
    double saved = 1.0 - 1.0 / speedup;   /* fraction of compute eliminated */
    if (saved < 0) saved = 0;
    if (saved >= 1.0) saved = 1.0 - 1e-9;
    return saved;
}

double wubu_pm_effective_capability(const wubu_postmoore_t *p) {
    if (!p) return 0.0;
    return wubu_pm_total_speedup(p);   /* same silicon does N-x more */
}

double wubu_pm_algorithmic_years(const wubu_postmoore_t *p) {
    if (!p) return 0.0;
    double speedup = wubu_pm_total_speedup(p);
    if (speedup < 1.0) return 0.0;
    return log2(speedup) * 2.0;   /* Moore doubles every ~2 years */
}
