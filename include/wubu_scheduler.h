#ifndef WUBU_SCHEDULER_H
#define WUBU_SCHEDULER_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void wubu_sched_init(int max_batch, int policy);
int  wubu_sched_submit(int id, int n_tokens);
int  wubu_sched_next(int ids_out[/*max_ids*/], int max_ids);
void wubu_sched_complete(int id);
int  wubu_sched_active(void);

enum { SCHED_FIFO = 0, SCHED_ROUND_ROBIN = 1 };

#ifdef __cplusplus
}
#endif
#endif
