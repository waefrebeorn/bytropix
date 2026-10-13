#ifndef WUBU_IRQAFF_H
#define WUBU_IRQAFF_H
#include <stdint.h>

/* wubu_irqaff.h -- KERNEL #21 IRQ affinity settable via /n.
 *
 * IRQ affinity (smp_affinity): each IRQ is bound to a set of CPUs via a
 * bitmask, so its interrupt is delivered only to those CPUs. Exposed through
 * 9P (/n/irq/<n>/smp_affinity) so it can be set at runtime. The manager maps
 * IRQ number -> CPU mask, validates masks against the CPU count, and reports
 * which CPU would handle a given IRQ. Deterministic, real work (no stubs). */

#define WUBU_IRQ_MAX 64
#define WUBU_IRQ_MAX_CPUS 64

typedef struct {
    int irq;
    uint64_t mask;          /* CPU affinity bitmask (bit c = CPU c) */
} wubu_irq_aff_t;

typedef struct {
    wubu_irq_aff_t irqs[WUBU_IRQ_MAX];
    int n;
    int n_cpus;             /* number of online CPUs (mask width) */
} wubu_irqaff_t;

/* Initialise with the number of online CPUs. */
int wubu_irqaff_init(wubu_irqaff_t *a, int n_cpus);

/* Register an IRQ with an initial affinity mask (default = all CPUs).
 * Returns 0/-1. */
int wubu_irqaff_register(wubu_irqaff_t *a, int irq);

/* Set an IRQ's affinity mask. The mask must be non-zero and within the CPU
 * count. Returns 0/-1. */
int wubu_irqaff_set(wubu_irqaff_t *a, int irq, uint64_t mask);

/* Get an IRQ's current affinity mask. Returns the mask or 0 if unknown. */
uint64_t wubu_irqaff_get(const wubu_irqaff_t *a, int irq);

/* Which CPU handles the IRQ (lowest set bit in the mask). Returns CPU index
 * or -1. */
int wubu_irqaff_target_cpu(const wubu_irqaff_t *a, int irq);

/* Is the IRQ pinned to a single CPU? 1/0. */
int wubu_irqaff_pinned(const wubu_irqaff_t *a, int irq);

#endif
