#ifndef WUBU_MEGA_H
#define WUBU_MEGA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_mega wubu_mega_t;

/* MEGA single step: EMA state + LSTM gates + gated attention fusion. */
wubu_mega_t *wubu_mega_create(int d_model, int d_ema);
void wubu_mega_free(wubu_mega_t *m);
void wubu_mega_step(const wubu_mega_t *m, const float *x, float *state,
                    float forget_l, float input_l, float gate_l,
                    const float *attn_out, float *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MEGA_H */
