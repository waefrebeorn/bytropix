/* wubu_irqaff.c -- KERNEL #21 IRQ affinity settable via /n.
 * See wubu_irqaff.h. */
#include "wubu_irqaff.h"
#include <stddef.h>

int wubu_irqaff_init(wubu_irqaff_t *a, int n_cpus) {
    if (!a || n_cpus <= 0 || n_cpus > WUBU_IRQ_MAX_CPUS) return -1;
    a->n = 0;
    a->n_cpus = n_cpus;
    return 0;
}

int wubu_irqaff_register(wubu_irqaff_t *a, int irq) {
    if (!a || irq < 0 || a->n >= WUBU_IRQ_MAX) return -1;
    /* avoid duplicate registration */
    for (int i = 0; i < a->n; i++) if (a->irqs[i].irq == irq) return -1;
    a->irqs[a->n].irq = irq;
    /* default: all online CPUs */
    uint64_t all = (a->n_cpus >= 64) ? ~0ULL : ((1ULL << a->n_cpus) - 1);
    a->irqs[a->n].mask = all;
    a->n++;
    return 0;
}

static wubu_irq_aff_t *find(wubu_irqaff_t *a, int irq) {
    for (int i = 0; i < a->n; i++) if (a->irqs[i].irq == irq) return &a->irqs[i];
    return NULL;
}

int wubu_irqaff_set(wubu_irqaff_t *a, int irq, uint64_t mask) {
    if (!a || mask == 0) return -1;   /* empty affinity is invalid */
    uint64_t valid = (a->n_cpus >= 64) ? ~0ULL : ((1ULL << a->n_cpus) - 1);
    if (mask & ~valid) return -1;     /* references an offline CPU */
    wubu_irq_aff_t *e = find(a, irq);
    if (!e) return -1;                /* unknown IRQ */
    e->mask = mask;
    return 0;
}

uint64_t wubu_irqaff_get(const wubu_irqaff_t *a, int irq) {
    if (!a) return 0;
    for (int i = 0; i < a->n; i++) if (a->irqs[i].irq == irq) return a->irqs[i].mask;
    return 0;
}

int wubu_irqaff_target_cpu(const wubu_irqaff_t *a, int irq) {
    if (!a) return -1;
    uint64_t m = wubu_irqaff_get(a, irq);
    if (m == 0) return -1;
    int cpu = 0;
    while (!(m & 1)) { m >>= 1; cpu++; }
    return cpu;
}

int wubu_irqaff_pinned(const wubu_irqaff_t *a, int irq) {
    if (!a) return 0;
    uint64_t m = wubu_irqaff_get(a, irq);
    if (m == 0) return 0;
    return (m & (m - 1)) == 0 ? 1 : 0;   /* exactly one bit set */
}
