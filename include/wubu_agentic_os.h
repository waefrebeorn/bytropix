/*
 * wubu_agentic_os.h -- AGI-OS agentic runtime governance (AD01-AD04).
 */
#ifndef WUBU_AGENTIC_OS_H
#define WUBU_AGENTIC_OS_H

typedef struct { long seq; int step; } wubu_checkpoint_t;

typedef struct {
    long cpu_ms_max;
    long ram_mb_max;
    long io_kb_max;
} wubu_resbound_t;

/* AD01 9P capability enforcement: is path allowed under agent_subtree? */
int wubu_9p_cap_allowed(const char *agent_subtree, const char *path);
/* AD02 exponential backoff (ms). */
long wubu_backoff_ms(int attempt, long base, int cap);
/* AD02 skip-if-running. */
int wubu_skip_if_running(int running_flag);
/* AD03 checkpoint pack/resume. */
void wubu_checkpoint_pack(wubu_checkpoint_t *c, long seq, int step);
int  wubu_checkpoint_resume(const wubu_checkpoint_t *c, long *seq, int *step);
/* AD04 resource budget check (0=ok, else bitmask of overrun dims). */
int  wubu_resbound_check(const wubu_resbound_t *b, long cpu_ms, long ram_mb, long io_kb);

#endif
