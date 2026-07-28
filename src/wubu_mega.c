/*
 * wubu_mega.c — MEGA: Moving Average Equipped Gated Attention (Round-3 #233/#235/#236).
 * C11, self-contained. MEGA = single-head gated attention + multi-headed EMA
 * state with LSTM-style input/forget gates. This implements one EMA+gated step:
 *   state = forget * state + input_gate * x        (EMA recurrence)
 *   gated = sigmoid(g) * (attn(x) + state_proj)     (gated fusion)
 * Used by BTL-3 (MEGA architecture, not vanilla transformer).
 */
#include "wubu_mega.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

struct wubu_mega {
    int d_ema;     /* EMA state channels */
    int d_model;   /* hidden dim */
};

wubu_mega_t *wubu_mega_create(int d_model, int d_ema) {
    if (d_model <= 0 || d_ema <= 0) return NULL;
    wubu_mega_t *m = (wubu_mega_t *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->d_model = d_model; m->d_ema = d_ema;
    return m;
}
void wubu_mega_free(wubu_mega_t *m) { free(m); }

/* One EMA+gated step. state (d_ema) updated in place. Returns fused output (d_model).
 * x: input d_model. forget_gate/input_gate/gate: scalars (pre-sigmoid logits ok:
 * we apply sigmoid inside). attn_out: d_model (attention result for this step). */
void wubu_mega_step(const wubu_mega_t *m, const float *x, float *state,
                    float forget_l, float input_l, float gate_l,
                    const float *attn_out, float *out) {
    float f = 1.0f / (1.0f + expf(-forget_l));   /* forget gate */
    float ig = 1.0f / (1.0f + expf(-input_l));   /* input gate */
    float g = 1.0f / (1.0f + expf(-gate_l));     /* output gate */
    /* EMA recurrence over d_ema channels (use first d_ema of x as EMA input). */
    int e = m->d_ema;
    for (int i = 0; i < e; i++)
        state[i] = f * state[i] + ig * x[i];
    /* Fuse attention output with projected EMA state. */
    for (int i = 0; i < m->d_model; i++) {
        float s_proj = (i < e) ? state[i] : 0.0f;
        out[i] = g * (attn_out[i] + s_proj);
    }
}
