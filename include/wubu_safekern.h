/*
 * wubu_safekern.h -- AGI-OS safety kernel (AF11-AF13).
 */
#ifndef WUBU_SAFEKERN_H
#define WUBU_SAFEKERN_H

#define WUBU_OOM_CEILING 524288  /* 512K invariant */

typedef struct {
    int stop_flag;     /* kernel-owned; reasoner cannot set/clear */
    int oom_ceiling;   /* externalized constraint (immutable by RSI) */
    int gate_enabled;  /* hard OOM gate always on */
} wubu_safekern_t;

enum { WUBU_CONT_NONE=0, WUBU_CONT_WARN, WUBU_CONT_THROTTLE,
       WUBU_CONT_SUSPEND, WUBU_CONT_STOP };

int  wubu_stop_honored(const wubu_safekern_t *k);              /* AF11 */
int  wubu_containment_level(float severity);                   /* AF12 */
int  wubu_containment_reversible(int level);                   /* AF12 */
int  wubu_rsi_mutation_ok(const wubu_safekern_t *k,
                          int proposed_max_ctx, int gate_enabled); /* AF13 */

#endif
